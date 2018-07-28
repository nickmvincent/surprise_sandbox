"""
This is just a convenience script
that runs process_results with a bunch
of saved params.
"""


import os
import pandas as pd
import glob
from shutil import copyfile


# specify where the primary results are located
# default (i.e. if you run the scripts and don't manually move anything) is the "results" directory


def main():
    d = "ml-20m_collected"
    for subdir in [
        "results", "processed_results",
    ]:
        path = '{}/{}'.format(d, subdir)
        if not os.path.exists(path):
            print('Missing directory {}, going to create it.'.format(path))
            os.makedirs(path)

    final_df = None
    outnames = []

    root_dir = 's3/ml-20m_autogen_aws_1,10/'
    files = glob.iglob(root_dir + '*/out/results/*.csv')
    for filepath in files:
        print(filepath)
        filename = os.path.basename(filepath)
        print('filename', filename)
        copyfile(filepath, d + '/results/{}'.format(filename))

    
    files = os.listdir(d + '/results')

    for file in files:
        outnames.append(d + "/results/" + file)
    os.system("python process_results.py --outname {}".format(','.join(outnames)))

    for file in files:
        print(file)
        try:
            df = pd.read_csv(d + '/processed_results/' + file)
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
    final_df[cols].to_csv(d + '/all_results.csv')

    # now you're ready to visualize and the interpret the results!

if __name__ == '__main__':
    main()