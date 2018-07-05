# Surprise Sandbox
Able to run a variety of recsys tests using a fork of [Surprise](http://surpriselib.com/). The fork lives [here](www.google.com).
Requires Jupyter, Pandas, numpy, seaborn, scipy, and matplotlib to explore results.
Recommended approach is to just install latest Anaconda distribution.

# Directory Setup
There are some directories required which are not tracked by git (because they contain a lot of large files that are generated throughout the experiment pipeline).

/predictions - predictions go here (user, movie, predicted rating, real rating) for each algo/boycott instance/fold. To use saving and loading of predictions, you need to set a constant random seed for shuffle base cross folds (currently set as random_state=0, so if you don't change anything it will work).
Subdirs of /predictions:
  /standards
  /boycotts

/results - raw results, e.g. "this algorithm had an RMSE of 0.9 under some boycott conditions"
/processed_results - computed additive and percent differences, e.g. "this algorithms had a 5% decrease in NDCG10 compared to no-boycott"
/standard_results - standard, no-boycott results here.

These will be populated when you run sandbox.py

After a run of sandbox.py, you'll get prediction files in /predictions, performance measure results in /results, a list of which users participated in each simulated boycott in /standard_results/uid_sets_*, and the standard, no-boycott results for each algorithm (specified in specs.py) in /standard_results.



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
cli, finder/explorer app, aws cli, etc

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

# Output Files
Currently, the experiement produce outputs that are written directly into files (as opposed to storing results in a database).