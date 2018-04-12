"""
Computes a percentile for each movie/item in the dataset
"""
from collections import defaultdict, Counter
import argparse

import pandas as pd
from utils import get_dfs

def main(args):
    """driver"""
    dfs = get_dfs(args.dataset)
    ratings_df = dfs['ratings']
    movies_df = dfs['movies']

    d = defaultdict(int)
    for i, row in ratings_df.iterrows():
        d[row.movie_id] += 1
    
    c = Counter(d)
    top_five_percent = c.most_common(len(d) // 20)
    print(top_five_percent)
    movie_ids = [movie_id[0] for movie_id in top_five_percent]
    with open('boycott_files/{}_top_five_percent_movies.csv'.format(args.dataset), 'w') as outfile:
        outfile.write(','.join([str(x) for x in movie_ids]))
    for movie_id in movie_ids:
        print(movies_df[movies_df.movie_id == movie_id].movie_title)
    

def parse():
    """
    Parse args and handles list splitting

    Example:
    python process_results --sample_sizes 4,5,6,7 --num_samples 100
    python process_results --grouping gender --userfracs 0.5,1 --ratingfracs 0.5,1
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ml-1m')
    args = parser.parse_args()
    main(args)

if __name__ == '__main__':
    parse()