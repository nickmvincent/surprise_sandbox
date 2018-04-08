"""
Process results

Currently this just means computing metric increases,
in both additive units and percent units.
"""


import argparse
import json

import pandas as pd
import numpy as np

from utils import concat_output_filename
from plot import plot_all

def main(args):
    algo_names = [
        'SVD',
        'KNNBaseline_item_msd',
    ]
    measures = ['RMSE', 'MAE', 'prec10t4_prec5t4_rec10t4_rec5t4_ndcg10_ndcg5_ndcgfull']
    metric_names = []
    for measure in measures:
        if '_' in measure:
            metric_names += measure.lower().split('_')
        else:
            metric_names.append(measure.lower())
    standard_results = {}    
    for userfrac in args.userfracs:
        for ratingfrac in args.ratingfracs:
            for sample_size in args.sample_sizes:
                outname = concat_output_filename(
                    args.dataset, args.grouping,
                    userfrac,
                    ratingfrac,
                    sample_size, args.num_samples
                )
                try:
                    err_df = pd.read_csv(outname)
                except FileNotFoundError:
                    print('skipping ' + outname)
                    continue
                err_df = err_df.set_index('Unnamed: 0')
                uid_to_metric = err_df.to_dict(orient='index')
                for algo_name in algo_names:
                    filename_ratingcv_standards = 'standard_results/{}_ratingcv_standards_for_{}.json'.format(
                        args.dataset, algo_name)
                    with open(filename_ratingcv_standards, 'r') as f:
                        standard_results[algo_name] = json.load(f)
                    for uid, res in uid_to_metric.items():
                        for metric in metric_names:
                            standard_val = standard_results[algo_name].get(metric)
                            if standard_val is None:
                                continue
                            for group in ['all', 'non-boycott', 'boycott', 'like-boycott', 'all-like-boycott']:
                                key = '{}_{}'.format(metric, group)
                                vals = res.get(key)
                                print(vals)
                                if vals:
                                    meanval = np.mean(vals)
                                    old_add_inc_key = 'increase_from_baseline_{}'.format(key)
                                    new_add_inc_key = 'increase_{}'.format(key)
                                    add_inc = res.get(old_add_inc_key)
                                    add_inc_computed = meanval - standard_val

                                    per_inc_key = 'percent_increase_{}'.format(key)
                                    per_inc_computed = 100 * (meanval - standard_val) / standard_val
                                    if add_inc:
                                        print(add_inc, add_inc_computed)
                                        # assert(add_inc == add_inc_computed)
                                    uid_to_metric[uid][new_add_inc_key] = add_inc_computed
                                    uid_to_metric[uid][per_inc_key] = per_inc_computed
                pd.DataFrame.from_dict(uid_to_metric, orient='index').to_csv(outname.replace('results/', 'processed_results/'))

                            

def parse():
    """
    Parse args and handles list splitting

    Example:
    python process_results --sample_sizes 4,5,6,7 --num_samples 100
    python process_results --grouping gender --userfracs 0.5,1 --ratingfracs 0.5,1
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--grouping', default='sample_users')
    parser.add_argument('--sample_sizes')
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--dataset', default='ml-1m')
    parser.add_argument('--userfracs')
    parser.add_argument('--ratingfracs')
    args = parser.parse_args()
    if args.sample_sizes:
        args.sample_sizes = [int(x) for x in args.sample_sizes.split(',')]
        if args.num_samples is None:
            args.num_samples = 1000
    else:
        args.sample_sizes = [None]

    if args.userfracs:
        args.userfracs = [float(x) for x in args.userfracs.split(',')]
    else:
        args.userfracs = [1.0]
    if args.ratingfracs:
        args.ratingfracs = [float(x) for x in args.ratingfracs.split(',')]
    else:
        args.ratingfracs = [1.0]

    if args.grouping == 'sample':
        args.grouping = 'sample_users'
    main(args)


if __name__ == '__main__':
    parse()
