import argparse

import pandas as pd
import matplotlib.pyplot as plt

from utils import concat_output_filename
from plot import plot_all

def main(args):
    for sample_size in args.sample_sizes:
        outname = concat_output_filename(
            args.dataset, args.grouping,
            args.userfrac,
            args.ratingfrac,
            sample_size, args.num_samples
        )

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
    parser.add_argument('--show_plots', action='store_true')
    parser.add_argument('--userfrac', type=float, default=1.0)
    parser.add_argument('--ratingfrac', type=float, default=1.0)
    args = parser.parse_args()
    if args.sample_sizes:
        args.sample_sizes = [int(x) for x in args.sample_sizes.split(',')]
        if args.num_samples is None:
            args.num_samples = 1000
    else:
        args.sample_sizes = [None]
    if args.grouping == 'sample':
        args.grouping = 'sample_users'
    main(args)


if __name__ == '__main__':
    parse()
