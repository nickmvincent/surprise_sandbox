"""
This module is for plotting the errors

To use this, first run sandbox.py to generate data in csv format
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd


NCOLS = 3

def main(args):
    """Do the plotting"""
    if args.mode == 'hist':
        df = pd.DataFrame.from_csv(args.f)
        cols = df.columns.values
        print(len(cols))
        subplot_y_max = (len(cols) + 1) // NCOLS

        fig, axes = plt.subplots(nrows=subplot_y_max, ncols=NCOLS)
        fig.suptitle(args.f)
        for i_col, col in enumerate(df.columns.values):
            subplot_x = (i_col) % NCOLS
            subplot_y = (i_col) // NCOLS
            print(subplot_y, subplot_x)
            ax = axes[subplot_y, subplot_x]
            
            X = df[col]
            ax.set_title(col)
            X.plot.hist(ax=ax, alpha=0.5)
            ax.axvline(X.mean(), color='b', linestyle='dashed', linewidth=2)


    plt.show()


def parse():
    """
    Parse args and handles list splitting
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='hist')
    parser.add_argument('--f', default='err_df-sample_users-250.csv')
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    parse()