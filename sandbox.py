"""
A flexible script for exploring performance of the
Surprise recommender system library
using various MovieLens datasets
"""
# pylint: disable=E0401
import argparse
import json
import time
from collections import OrderedDict, defaultdict
import os
import datetime
import sys

import pandas as pd
import numpy as np
from utils import get_dfs, concat_output_filename, load_head_items
from specs import ALGOS, ALGOS_FOR_STANDARDS, NUM_FOLDS
from constants import MEASURES
from prep_organized_boycotts import (
    group_by_age, group_by_gender, group_by_genre,
    group_by_occupation, group_by_power, group_by_state, group_by_genre_strict
)
from joblib import Parallel, delayed

from surprise.model_selection import cross_validate_custom
from surprise import SVD, Dataset, KNNBaseline, GuessThree, GlobalMean, MovieMean
from surprise.reader import Reader

# long-term: massively abstract this code so it will work w/ non-recsys algorithsm


def task(
    algo_name, algo, nonboycott, boycott, boycott_uid_set,
    like_boycotters_uid_set, measures, cv, verbose, identifier,
    num_ratings, num_users, num_movies, name, head_items, save_path):
    return {
        'subset_results': cross_validate_custom(
            algo, nonboycott, boycott, boycott_uid_set,
            like_boycotters_uid_set, measures, cv, n_jobs=1,
            head_items=head_items, save_path=save_path
        ),
        'num_ratings': num_ratings,
        'num_users': num_users,
        'num_movies': num_movies,
        'name': name,
        'algo_name': algo_name,
        'identifier': identifier
    }


def main(args):
    """
    Run the sandbox experiments
    """
    # TODO: support FLOAT ratings for ml-20m... only supports int right now!
    times = OrderedDict()
    times['start'] = time.time()
    algos = ALGOS
    if args.movie_mean:
        algos = {
            'MovieMean': MovieMean(),
            'GlobalMean': GlobalMean(),
        }
    algos_for_standards = ALGOS_FOR_STANDARDS
    dfs = get_dfs(args.dataset)
    head_items = load_head_items(args.dataset)
    times['dfs_loaded'] = time.time() - times['start']
    print('Got dataframes, took {} seconds'.format(times['dfs_loaded']))
    print('Total examples: {}'.format(len(dfs['ratings'].index)))

    ratings_df, users_df, movies_df = dfs['ratings'], dfs['users'], dfs['movies']
    if args.mode == 'info':
        print(ratings_df.memory_usage(index=True))
        print(users_df.memory_usage(index=True))
        print(movies_df.memory_usage(index=True))

        print(ratings_df.info())
        print(users_df.info())
        return
    data = Dataset.load_from_df(
        ratings_df[['user_id', 'movie_id', 'rating']],
        reader=Reader()
    )
    times['data_constructed'] = time.time() - times['dfs_loaded']

    # note to reader: why are precision, recall, and ndcg all stuffed together in one string?
    # this ensures they will be computed all at once. Evaluation code will split them up for presentation
    metric_names = []
    for measure in MEASURES:
        if '_' in measure:
            splitnames = measure.lower().split('_')
            metric_names += splitnames
            metric_names += [x + '_frac' for x in splitnames]
            metric_names += ['tail' + x for x in splitnames]
        else:
            metric_names.append(measure.lower())
    
    if args.compute_standards:
        standard_results = defaultdict(list)
        for algo_name in algos_for_standards:
            for _ in range(args.num_standards):        
                filename_ratingcv_standards = 'standard_results/{}_ratingcv_standards_for_{}.json'.format(
                    args.dataset, algo_name)

                print('Computing standard results for {}'.format(algo_name))
                save_path = os.getcwd() + '/predictions/standards/{}_{}_'.format(args.dataset, algo_name)

                results = cross_validate_custom(
                    algos_for_standards[algo_name], data, Dataset.load_from_df(pd.DataFrame(),
                    reader=Reader()), [], [], MEASURES, NUM_FOLDS, head_items=head_items,
                    save_path=save_path)
                saved_results = {}
                for metric in metric_names:
                    saved_results[metric] = np.mean(results[metric + '_all'])
                    frac_key = metric + '_frac_all'
                    if frac_key in results:
                        saved_results[frac_key] = np.mean(results[frac_key])

                with open(filename_ratingcv_standards, 'w') as f:
                    json.dump(saved_results, f)
                    
                standard_results[algo_name].append(saved_results)
            standard_results_df = pd.DataFrame(standard_results[algo_name])
            print(standard_results_df.mean())
            standard_results_df.mean().to_csv('{}'.format(
                filename_ratingcv_standards).replace('.csv', '_{}.csv'.format(
                    args.num_standards)
                )
            )

    times['standards_loaded'] = time.time() - times['data_constructed']

    experiment_configs = []
    if args.grouping == 'individual_users':
        experiment_configs += [{'type': 'individual_users', 'size': None}]
    elif args.grouping == 'sample':
        if args.sample_sizes:
            experiment_configs += [
                {
                    'type': 'sample_users', 'size': sample_size
                } for sample_size in args.sample_sizes]
        else:
            raise ValueError(
                'When using grouping="sample", you must provide a set of sample sizes')
    elif args.grouping in [
        'gender', 'age', 'power', 'state', 'genre', 'genre_strict', 'occupation', 
    ]:
        experiment_configs += [{'type': args.grouping, 'size': None}]
    else:
        experiment_configs = []

    num_configs = len(experiment_configs)
    if args.sample_sizes:
        num_runs = num_configs * args.num_samples
        print('{} total train/tests will be run because you chose {} sample_sizes and number of samples of {}'.format(
            num_runs, num_configs, args.num_samples
        ))
    else:
        print('{} total train/tests will be will be run'.format(num_configs))
        num_runs = num_configs
    secs = 47 * num_runs
    hours = secs / 3600
    time_estimate = """
        At a rate of 47 seconds per run, this should take: {} seconds ({} hours)
    """.format(secs, hours)
    print(time_estimate)
    uid_to_error = {}
    experimental_iterations = []
    for config in experiment_configs:
        outname = concat_output_filename(
            args.dataset, config['type'], args.userfrac,
            args.ratingfrac,
            config['size'], args.num_samples, args.indices
        )
        if config['type'] == 'individual_users':
            experimental_iterations = list(users_df.iterrows())
        elif config['type'] == 'sample_users':
            experimental_iterations = [{
                'df': users_df.sample(config['size']), # copies user_df
                'name': '{} user sample'.format(config['size'])
            } for _ in range(args.num_samples)]
        elif config['type'] == 'gender':
            for _ in range(args.num_samples):
                experimental_iterations += group_by_gender(users_df)
        elif config['type'] == 'age':
            for _ in range(args.num_samples):
                experimental_iterations += group_by_age(users_df)
        elif config['type'] == 'state':
            for _ in range(args.num_samples):
                experimental_iterations += group_by_state(users_df, dataset=args.dataset)
        elif config['type'] == 'genre':
            for _ in range(args.num_samples):
                experimental_iterations += group_by_genre(
                    users_df=users_df, ratings_df=ratings_df, movies_df=movies_df,
                    dataset=args.dataset)
        elif config['type'] == 'genre_strict':
            for _ in range(args.num_samples):
                experimental_iterations += group_by_genre_strict(
                    users_df=users_df, ratings_df=ratings_df, movies_df=movies_df,
                    dataset=args.dataset)
        elif config['type'] == 'power':
            for _ in range(args.num_samples):
                experimental_iterations += group_by_power(users_df=users_df, ratings_df=ratings_df, dataset=args.dataset)
        elif config['type'] == 'occupation':
            for _ in range(args.num_samples):
                experimental_iterations += group_by_occupation(users_df)

        experiment_identifier_to_uid_sets = defaultdict(lambda: defaultdict(list))
        for algo_name in algos:
            delayed_iteration_list = []
            for i, experimental_iteration in enumerate(experimental_iterations):
                if config['type'] == 'individual_users':
                    row = experimental_iteration[1]
                    identifier = row.user_id
                    name = 'individual'
                    if args.indices != 'all':
                        if identifier < args.indices[0] or identifier > args.indices[1]:
                            continue
                    boycott_uid_set = set([row.user_id])
                    like_boycotters_uid_set = set([])
                
                # elif config['type'] == 'sample_users' and config['size'] == 0:
                #     identifier = i
                #     name = experimental_iteration['name']
                #     boycott_uid_set = set([])
                #     like_boycotters_uid_set = set([])

                elif config['type'] in [
                    'sample_users',
                    'gender', 'age', 'power', 'state', 'genre',
                    'genre_strict',
                    'occupation',
                ]:
                    identifier = i
                    name = experimental_iteration['name']

                    possible_boycotters_df = experimental_iteration['df']                        
                    print(name)
                    print(possible_boycotters_df.head())
                    if args.userfrac != 1.0:
                        boycotters_df = possible_boycotters_df.sample(frac=args.userfrac)
                    else:
                        boycotters_df = possible_boycotters_df
                    boycott_uid_set = set(boycotters_df.user_id)
                    like_boycotters_df = possible_boycotters_df.drop(boycotters_df.index)
                    like_boycotters_uid_set = set(like_boycotters_df.user_id)

                non_boycott_user_ratings_df = ratings_df[~ratings_df.user_id.isin(boycott_uid_set)] # makes a df copy
                boycott_ratings_df = None
                boycott_user_lingering_ratings_df = None
                for uid in boycott_uid_set:
                    ratings_belonging_to_user = ratings_df[ratings_df.user_id == uid]
                    if args.ratingfrac != 1.0:
                        boycott_ratings_for_user = ratings_belonging_to_user.sample(frac=args.ratingfrac)
                    else:
                        boycott_ratings_for_user = ratings_belonging_to_user
                    lingering_ratings_for_user = ratings_belonging_to_user.drop(boycott_ratings_for_user.index)
                    if boycott_ratings_df is None:
                        boycott_ratings_df = boycott_ratings_for_user
                    else:
                        boycott_ratings_df = pd.concat([boycott_ratings_df, boycott_ratings_for_user])
                    if boycott_user_lingering_ratings_df is None:
                        boycott_user_lingering_ratings_df = lingering_ratings_for_user
                    else:
                        boycott_user_lingering_ratings_df = pd.concat([boycott_user_lingering_ratings_df, lingering_ratings_for_user])
                print('Iteration: {}'.format(i))
                print('  Boycott ratings: {}, Lingering Ratings from Boycott Users: {}'.format(
                    len(boycott_ratings_df.index), len(boycott_user_lingering_ratings_df.index)
                ))
                all_non_boycott_ratings_df = pd.concat(
                    [non_boycott_user_ratings_df, boycott_user_lingering_ratings_df])

                print('  non_boycott_user_ratings_df.info()', non_boycott_user_ratings_df.info())
                print('  all_non_boycott_ratings_df.info()', all_non_boycott_ratings_df.info())

                nonboycott = Dataset.load_from_df(
                    all_non_boycott_ratings_df[['user_id', 'movie_id', 'rating']],
                    reader=Reader()
                ) # makes a copy
                boycott = Dataset.load_from_df(
                    boycott_ratings_df[['user_id', 'movie_id', 'rating']],
                    reader=Reader()
                ) # makes a copy
                print('  nonboycott Dataset obj: {}\n  boycott Dataset obj:{}'.format(sys.getsizeof(nonboycott), sys.getsizeof(boycott)))

                identifier = str(identifier).zfill(4)
                num_users = len(all_non_boycott_ratings_df.user_id.value_counts())
                num_movies = len(all_non_boycott_ratings_df.movie_id.value_counts())
                num_ratings =  len(all_non_boycott_ratings_df.index)

                # make sure to save the set of boycott ids and like boycott ids
                experiment_identifier_to_uid_sets[identifier]['boycott_uid_set'] = ';'.join(str(x) for x in boycott_uid_set)
                experiment_identifier_to_uid_sets[identifier]['like_boycotters_uid_set'] = ';'.join(str(x) for x in like_boycotters_uid_set)
                save_path = outname.replace('results/', '/predictions/boycotts/{}__'.format(identifier)).replace('.csv', '_')
                save_path = os.getcwd() + save_path
                delayed_iteration_list += [delayed(task)(
                    algo_name, algos[algo_name], nonboycott, boycott, boycott_uid_set, like_boycotters_uid_set, MEASURES, NUM_FOLDS,
                    False, identifier,
                    num_ratings,
                    num_users,
                    num_movies, name,
                    head_items, save_path=save_path,
                )]

            # data should be ~30 MB
            print('About to run Parallel()')
            out_dicts = Parallel(n_jobs=-1)(tuple(delayed_iteration_list))
            for d in out_dicts:
                res = d['subset_results']
                algo_name = d['algo_name']
                uid = str(d['identifier']) + '_' + d['algo_name']
                uid_to_error[uid] = {
                    'num_ratings': d['num_ratings'],
                    'num_users': d['num_users'],
                    'num_movies': d['num_movies'],
                    'name': d['name'],
                    'algo_name': d['algo_name'],
                }
                for metric in metric_names + ['fit_time', 'test_times', 'num_tested']:
                    for group in ['all', 'non-boycott', 'boycott', 'like-boycott', 'all-like-boycott']:
                        key = '{}_{}'.format(metric, group)
                        # if group in ['boycott', ]:
                        #     val = np.nanmean(res[key])
                        vals = res.get(key)
                        if vals:
                            val = np.mean(res[key])
                            uid_to_error[uid].update({
                                key: val,
                            })
        err_df = pd.DataFrame.from_dict(uid_to_error, orient='index')
        
        uid_sets_outname = outname.replace('results/', 'standard_results/uid_sets_')
        pd.DataFrame.from_dict(experiment_identifier_to_uid_sets, orient='index').to_csv(uid_sets_outname)
        if args.movie_mean:
            outname = outname.replace('results/', 'results/MOVIEMEAN_')
        err_df.to_csv(outname)
        print('Full runtime was: {} for {} runs'.format(time.time() - times['start'], num_runs))


def parse():
    """
    Parse args and handles list splitting

    Example: 
    python sandbox.py --grouping state

    python sandbox.py --grouping sample --sample_sizes 3 --num_samples 2 --dataset test_ml-1m --compute_standards
    python sandbox.py --grouping sample --sample_sizes 1 --num_samples 10 --dataset ml-20m --indices 1,10
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--indices', default='all')
    parser.add_argument('--grouping')
    parser.add_argument('--sample_sizes')
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--dataset', default='ml-1m')
    parser.add_argument(
        '--compute_standards', action='store_true',
        help='Defaults to false. Pass --compute_standards if you want to compute standards (you really only need to do this once)')
    parser.add_argument(
        '--num_standards', default=1, type=int,
        help='number of times to replicate standards calculation (to account for random fold shuffling)'
    )
    parser.add_argument(
        '--movie_mean', help='Defaults to False. If True, override everything and just use MovieMean and GlobalMean',
        action='store_true')
    parser.add_argument('--mode', default='compute')
    parser.add_argument('--userfrac', type=float, default=1.0)
    parser.add_argument('--ratingfrac', type=float, default=1.0)
    args = parser.parse_args()

    # make dirs as needed
    for directory in [
        'logs',
        'results',
        'standard_results',
        'processed_results',
        'predictions',
        'predictions/standards',
        'predictions/boycotts',
    ]:
        if not os.path.exists(directory):
            print('Missing directory {}, going to create it.'.format(directory))
            os.makedirs(directory)

    with open('logs/{}.txt'.format(datetime.date.today()), 'a') as f:
        msg = '{}\n{}\n\n'.format(str(datetime.datetime.now()), str(args))
        f.write(msg)

    if args.sample_sizes:
        args.sample_sizes = [int(x) for x in args.sample_sizes.split(',')]
        if args.num_samples is None:
            args.num_samples = 1000
    else:
        if args.num_samples is None:
            args.num_samples = 1

    if ',' in args.indices:
        args.indices = [int(x) for x in args.indices.split(',')]

    main(args)


if __name__ == '__main__':
    parse()
