"""
Summarize results using prints and plots
"""
import argparse
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

from utils import concat_output_filename
from plot import plot_all

def main(args):
    across = defaultdict(lambda: defaultdict(list))
    algo_names = [
        'SVD',
        'KNNBaseline_item_msd',
    ]
    metrics = [
        'rmse', 'ndcg10',
        #'ndcg5', 'ndcgfull',
    ]
    for colunit in ['increase_{}', 'percent_increase_{}']:
        for userfrac in args.userfracs:
            for ratingfrac in args.ratingfracs:
                for sample_size in args.sample_sizes:
                    outname = 'processed_' + concat_output_filename(
                        args.dataset, args.grouping,
                        userfrac,
                        ratingfrac,
                        sample_size, args.num_samples
                    )
                    err_df = pd.read_csv(outname)

                    for algo_name in algo_names:
                        print('===\n' + algo_name)
                        filtered_df = err_df[err_df.algo_name == algo_name]
                        if args.verbose:
                            print(filtered_df.mean())
                        else:
                            colnames = []
                            for metric in metrics:
                                for group in args.test_groups:
                                    key = '{}_{}'.format(metric, group)
                                    colname = colunit.format(key)
                                    colnames.append(colname)
                            cols = filtered_df[colnames]
                            means = cols.mean()
                            print(means)
                            if args.plot_across:    
                                for col in cols.columns.values:
                                    across[algo_name][col].append(means[col])
                            if args.plot_histograms:
                                plot_all(cols, 'hist', algo_name + '_' + outname)
                    
                if args.plot_across:
                    for algo_name in algo_names:
                        _, axes = plt.subplots(ncols=len(metrics))
                        _, zoomaxes = plt.subplots(ncols=len(across[algo_name]))
                        metric_to_index = {}
                        for i_metric, metric in enumerate(metrics):
                            metric_to_index[metric] = i_metric
                        for i, (key, val) in enumerate(across[algo_name].items()):
                            for metric, index in metric_to_index.items():
                                if metric in key:
                                    i_metric = index
                            ax = axes[i_metric]
                            ax.plot(val)
                            ax.set_title(algo_name)
                            zoomax = zoomaxes[i]
                            zoomax.plot(val)
                            zoomax.set_title(algo_name + ' ' + key)
    plt.show()
    

def parse():
    """
    Parse args and handles list splitting

    example
    python summarize.py --sample_sizes 4,5,6,7 --num_samples 100 --plot_across
    python summarize.py --grouping gender --userfracs 0.5,1 --ratingfracs 0.5,1
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--grouping', default='sample_users')
    parser.add_argument('--sample_sizes')
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--dataset', default='ml-1m')
    parser.add_argument('--verbose')
    parser.add_argument('--plot_histograms', action='store_true')
    parser.add_argument('--plot_across', action='store_true')
    parser.add_argument('--userfracs')
    parser.add_argument('--ratingfracs')
    parser.add_argument('--test_groups', default='non-boycott,boycott')
    args = parser.parse_args()

    args.test_groups = args.test_groups.split(',')
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
    # for convenience
    if args.grouping == 'sample':
        args.grouping = 'sample_users'
    main(args)


if __name__ == '__main__':
    parse()
