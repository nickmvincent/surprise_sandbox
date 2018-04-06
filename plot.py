"""
This module is for plotting the errors

To use this, first run sandbox.py to generate data in csv format
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd


NCOLS = 3

def plot_all(df, mode='hist', name=''):
    """Do the plotting"""
    if mode == 'hist':
        cols = df.columns.values
        subplot_y_max = (len(cols) + 1) // NCOLS

        fig, axes = plt.subplots(nrows=subplot_y_max, ncols=NCOLS)
        fig.suptitle(name)
        for i_col, col in enumerate(df.columns.values):
            subplot_x = (i_col) % NCOLS
            subplot_y = (i_col) // NCOLS
            ax = axes[subplot_y, subplot_x]
            X = df[col]
            ax.set_title(col)
            X.plot.hist(ax=ax, alpha=0.5)
            ax.axvline(X.mean(), color='b', linestyle='dashed', linewidth=2)



def main():
    raise ValueError('implement this')

def parse():
    """
    Parse args and handles list splitting
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='hist')
    parser.add_argument('--f', default='indivs.csv')
    args = parser.parse_args()
    main()


if __name__ == '__main__':
    parse()