# all_results.csv readme

This CSV contains all processed data.

Most of the columns contain metric values, i.e.

prec10t4_non-boycott

or increases in metric values, e.g.
increase_prec10t4_non-boycott

The general format is
increase_{metric name}_{group name}

Percent increases have a "percent_" prepended, e.g.
percent_increase_prec10t4_non-boycott

Finally, there's also a set of columns for vanilla increases.
This is the performance difference compared to *the full user set under no boycott conditions"
e.g.
vanillapercent_increase_prec10t4_non-boycott

Using ratingfrac and userfrac columns to select the type of experiment you want.
If these cells are empty, that row was not properly processed (for example, KNN user-user was dropped halfway through so any KNN user-user rows are unprocessed)

An any way to deal w/ this in Pandas would be:
`df = df[df.ratingfrac.notna()]`

`algo_name` column can be used to identify which algorithm was used
`name` column indicates the type of experiment (e.g. female users boycott), while `type` identifies the broad category (e.g. gender boycott)

`num_users` indicates how many users were trained upon
So the number of users boycotting equal Total # of users (6040) minus the value in `num_users`
A variety of num_tested columns and `_times` columns exist to show exactly how many ratings were tested upon per crossfold and how long they took.