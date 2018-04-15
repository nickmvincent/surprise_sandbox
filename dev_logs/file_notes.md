# Data - where does it live?
## 4/14/2018
Currently, as there is a deadline coming up, the experiments are running in a variety of places: various screen windows on Butter and a whole lot of EC2 spot instances.

Therefore, tracking down data can be tricky. There are a variety of scripts meant to facilitate this (i.e. get the correct data files and don't accidentally copy old data files) but currently there is not a single convenient pipeline.

In this file, I'm going to lay out exactly where the data lives.
This should make it easier to figure out exactly which scrips ran with certain parameters in order to produce one output.

round6/ - this includes 10 samples (indices 1-10) user experiments for userfrac 0.5 and ratingfrac 0.5. Doesn't include state.

round7/ this includes 250 samples (indices 1-250) sample experiments. Includes 30, 60, 302, 604, ..., 5980.
However, some scripts ran too long and I manually cancelled them: 

Missing 1,3,6

round8/ - nothing. This errored out.

user_standards/ - this is where all the userfrac 0.5 standard calculations live

rand/ - this has 4832, 5436, and 5980

rand2/ - this has 1, 3, and 6

org01/ - this has an extra 10 samples (11-20) for gender, age, power, occupation, genre, genre_strict
It also has 20 samples (1-20) for state, as this was skipped previously.

full01/ - this has standards calculation for SVD full boycotts. Was missing KNNBaseline b/c it ran too long.
Was duplicated though, so each results folder has the same json file w/ standards.

full02 - this has standards calculation for all KNNBaseline full experiment (no genre_strict though)


TODO:
standards for sample experiments, just to check.
re-reun 11-20 (had issues w/ tail metrics)
standards for 0.5 userfrac experiments indices 11-20 (1-10 are done and included)