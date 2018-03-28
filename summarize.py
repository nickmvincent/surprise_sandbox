import argparse

import pandas as pd



def main(args):
    for sample_size in args.sample_sizes:
        outname = 'results/err_df-dataset_{}_type_{}-size_{}-sample_size_{}.csv'.format(
            args.dataset, args.grouping, sample_size,
            args.num_samples if args.num_samples else None)
        print(outname)
        err_df = pd.read_csv(outname)

        for algo_name in [
            'SVD',
            'KNNBaseline_item_msd',
        ]:
            print('===\n' + algo_name)
            filtered_df = err_df[err_df.algo_name == algo_name]
            if args.verbose:
                print(filtered_df.mean())
            else:
                colnames = []
                for metric in ['rmse', 'ndcg10']:
                    for group in ['all', 'in-group', 'out-group']:
                        key = '{}_{}'.format(metric, group)
                        colname = 'increase_from_baseline_{}'.format(key)
                        colnames.append(colname)
                print(filtered_df[colnames].mean())


def parse():
    """
    Parse args and handles list splitting
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--grouping', default='sample_users')
    parser.add_argument('--sample_sizes')
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--dataset', default='ml-1m')
    parser.add_argument('--verbose')
    args = parser.parse_args()
    if args.sample_sizes:
        args.sample_sizes = [int(x) for x in args.sample_sizes.split(',')]
        if args.num_samples is None:
            args.num_samples = 1000
    else:
        args.sample_sizes = [None]
    main(args)


if __name__ == '__main__':
    parse()
