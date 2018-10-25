"""
This is just a convenience script
that runs process_results with a bunch
of saved params.


This script does not have command line arguments, you need to edit the variables manually before running.
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

    for root_dir in [
        's3/ml-1m_autogen_aws_1,250/',
        's3/ml-1m_autogen_aws_1,10_grouped/',
        's3/ml-1m_autogen_aws_11,20_grouped/',
        's3/ml-1m_autogen_aws_21,30_grouped/',
        's3/ml-1m_autogen_aws_31,40_grouped/',
        's3/ml-1m_autogen_aws_41,50_grouped/',
    ]:
        files = glob.iglob(root_dir + '*/out/results/*.csv')
        for filepath in files:
            filename = os.path.basename(filepath)
            #print('filename', filename)
            copyfile(filepath, d + '/results/{}'.format(filename))

    
    files = os.listdir(d + '/results')


    for file in files:
        # weird pattern - why? Doing all the files at once is too long...
        outnames = []
        outnames.append(d + "/results/" + file)
        call = "python process_results.py --outname {}".format(','.join(outnames))
        print(call)
        os.system(call)

    for file in files:
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
    print("This script does not have command line arguments, you need to edit the variables manually before running.")
    main()