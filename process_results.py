"""
Process results

Currently this just means computing metric increases,
in both additive units and percent units.
"""


import argparse
import json


import pandas as pd
import numpy as np

from utils import concat_output_filename, extract_from_filename
from plot import plot_all

from constants import MEASURES, ALGO_NAMES, get_metric_names


def main(args):
    metric_names = get_metric_names()
    standard_results = {}
    outnames = []
    if args.outname:
        outnames.append('results/' + args.outname)
    else:
        for userfrac in args.userfracs:
            for ratingfrac in args.ratingfracs:
                for sample_size in args.sample_sizes:
                    outname = concat_output_filename(
                        args.dataset, args.grouping,
                        userfrac,
                        ratingfrac,
                        sample_size, args.num_samples
                    )
                    outnames.append(outname)
    for outname in outnames:

        userfrac = extract_from_filename(outname, 'userfrac-', 3)
        ratingfrac = extract_from_filename(outname, 'ratingfrac-', 3)
        experiment_type = extract_from_filename(outname, 'type-', None, '_userfrac')
        if 'indices' in outname:
            indices = extract_from_filename(outname, 'indices-', None, '.csv')
        print(outname)
        try:
            err_df = pd.read_csv(outname)
        except FileNotFoundError:
            continue
        err_df = err_df.set_index('Unnamed: 0')
        uid_to_metric = err_df.to_dict(orient='index')
        

        # this is dangerous. a lot of issues w/ accidental variable assignment
        for algo_name in ALGO_NAMES:
            standards_filename = 'MERGED_{}_{}.json'.format(args.dataset, algo_name)
            try:
                with open(standards_filename, 'r') as f:
                    standard_results = json.load(f)
            except:
                print('Could not load file {} for algo {}. Moving to next algorithm'.format(
                    standards_filename, algo_name
                ))
                standard_results = {}
                continue
            vanilla_filename = 'standard_results/{}_ratingcv_standards_for_{}.json'.format(args.dataset, algo_name)
            try:
                with open(vanilla_filename, 'r') as f:
                    vanilla = json.load(f)
            except:
                vanilla = {}
            for uid, res in uid_to_metric.items():
                if res['algo_name'] != algo_name:
                    continue
                for metric in metric_names:
                    for group in ['all', 'non-boycott', 'boycott', 'like-boycott', 'all-like-boycott']:
                        standard_val_key = '{metric}_{group}__{outname}__{identifier}'.format(**{
                            'metric': metric,
                            'group': group,
                            'outname': outname.replace('results/', ''),
                            'identifier': uid[:4], # the first 4 characters of the uid are a 4 digit identifier #
                        })
                        standard_val = standard_results.get(standard_val_key)
                        if standard_val is None:
                            print('Could not get standards with this key: {}, try vanilla'.format(standard_val_key))
                            if 'sample_users' in outname:
                                standard_val = vanilla.get(metric)
                                if standard_val is None:
                                    continue
                                else:
                                    print('Did load vanilla')
                            else:
                                continue
                        standard_val = np.mean(standard_val)
                        key = '{}_{}'.format(metric, group)
                        vals = res.get(key)
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
                            uid_to_metric[uid]['userfrac'] = userfrac
                            uid_to_metric[uid]['ratingfrac'] = ratingfrac
                            uid_to_metric[uid]['type'] = experiment_type
                            if 'indices' in outname:
                                uid_to_metric[uid]['indices'] = indices
        as_df = pd.DataFrame.from_dict(uid_to_metric, orient='index')
        
        cols = list(as_df.columns.values)
        for col in [
            'userfrac', 'ratingfrac', 'indices', 'name', 'algo_name', 'within_run_identifier', 'name'
        ]:
            if col in cols:
                cols.insert(0, cols.pop(cols.index(col)))
        as_df[cols].to_csv(outname.replace('results/', 'processed_results/'))

                            

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
    parser.add_argument('--outname')
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
