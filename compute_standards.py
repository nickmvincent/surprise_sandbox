"""
Computes baselines
"""
import os
import pandas as pd

from specs import ALGOS, ALGOS_FOR_STANDARDS


def main():
    files = os.listdir("uid_sets")
    delayed_iteration_list = []

    for file in files:

        delayed_iteration_list += [delayed(task)(
            algo_name, algos[algo_name], nonboycott, boycott, boycott_uid_set, like_boycotters_uid_set, measures, NUM_FOLDS,
            False, identifier,
            num_ratings,
            num_users,
            num_movies, name
        )]





if __name__ == '__main__'