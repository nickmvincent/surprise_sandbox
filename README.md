# Surprise Sandbox
Able to run a variety of recsys tests using Surprise (ADD LINK HERE)
Requires our forked version of Surprise to be installed to actually train and evaluate recommenders.
Requires Jupyter, Pandas, numpy, seaborn, scipy, and matplotlib to explore results.
Recommended approach is to just install latest Anaconda distribution.

# Benchmarks
See benchmark_comparisons.csv and http://surpriselib.com/.

# Inspecting Processed Data
This dump comes with processed results collected into `all_results.csv`
Therefore, if you load visualize-v02.py
(via `jupyter notebook` on the command line, or other method of your choice)

You can explore the various metrics, algorithms, etc.

In the 4th cell with code (directly below the cell that says "EDIT ME")
You can edit a variety of global variables to re-run the notebook with different configurations.

This is a quick way to compare various metrics, look at only at certain experiments, etc.


# Experiment Pipeline

Get files in places with your choice of:
cli, finder app, aws cli, etc

Be sure to install the forked version of surprise.

Organize all the "standards" (i.e. the results used for comparison with boycotts) into a single directory.
Here I've used misc_standards, and, and hard-coded that directory in `standards_for_uid_sets.py`

Merge standards files
`python .\standards_for_uid_sets.py --join`

Do the processing (i.e. match up columns and do substraction)
`python process_all.py`

Re-run visualization and/or statistics
`jupyter notebook`
Select "visualize-v02"