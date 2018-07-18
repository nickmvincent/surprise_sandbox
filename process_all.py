"""
This is just a convenience script
that runs process_results with a bunch
of saved params.
"""


import os
import pandas as pd


# specify where the primary results are located
# default (i.e. if you run the scripts and don't manually move anything) is the "results" directory
d = "results"
files = os.listdir(d)

final_df = None

outnames = []
for file in files:
    outnames.append(d + "/" + file)
os.system("python process_results.py --outname {}".format(','.join(outnames)))

for file in files:
    try:
        df = pd.read_csv('processed_results/' + file)
        if final_df is None:
            final_df = df
        else:
            final_df = pd.concat([final_df, df])
    except FileNotFoundError:
        print('File {} did not process'.format(file))


# put the cols into a consistent order for convenience
cols = list(final_df.columns.values)
for col in [
    'userfrac', 'ratingfrac', 'indices', 'name', 'algo_name', 'within_run_identifier', 'name'
]:
    if col in cols:
        cols.insert(0, cols.pop(cols.index(col)))
# specify a file for the final csv
final_df[cols].to_csv('all_results.csv')

# now you're ready to visualize and the interpret the results!