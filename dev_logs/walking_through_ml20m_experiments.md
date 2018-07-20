# Walking Through ML-20M Experiments

## Setup:
* download the surprise_sandbox library (this repo)
* download the forked Surprise library from Github and installed with `python setup.py install`.
* Activate my Anaconda environment and installed necessary libraries with pip. TODO - could make this easier

## Preparing batch files
* run script to generate a .sh file
`user_percents_to_user_counts.py` 


## Before running batch files
* Check "specs.py" and make sure you have selected the algorithms you're interested in
* If you wish to make experiments easier to replicate, be sure to set a random seed for SVD (hardcoded as 0 right now)
* Check "constants.py" and make sure the metrics you're interested in are included


## Compute standards
`python sandbox.py --dataset ml-20m --compute_standards --grouping none --indices 0,0`
Passing grouping=none will make it so only standards are computed.

## Run batch files
### On machine you control
`sh bash_scripts/ml-20m_autogen_aws.jobs.sh`

### With AWS EC2 Spot Instances
Copy jobs.txt to aws

`aws s3 cp prefix/jobs.txt s3://somepath/jobs.txt`

Run it

`sh aws/manage_aws.sh SECRET s3://somepath num_workers`

Wait until tasks finish and then download the results

`aws s3 cp s3://somepath/out destination/on/local/machine --recursive`

## Use the stored predictions to compute standard results for each perspective
* We will be using `standards_for_uid_sets.py`
* this script attempts to figure out all the "perspectives" that you need to get standards for, based on the collection of uid_sets files you've accrued throughout the experiments
* So we will copy all the uid_sets files of interest to some directory and pass it via the `pathto` argument. Or just use uid_sets/, which the is the default (you won't need to move anything)
* We can also use the `name_match` argument to grab only some of the uid_sets files from the `pathto` directory.
* Exactly how we compute these will depend on available computational resources

* Each boycott has an identifying key, like this
* "dataset-test_ml-1m_type-sample_users_userfrac-1.0_ratingfrac-1.0_sample_size-3_num_samples-2_indices-1-to-2.csv__0000"
* the 0000 is the identifer number, or index. This corresponds to iteration number. i.e. if this is 0009, this was the tenth iteration of a certain boycott size

* the `standards_for_uid_sets.py` will group these together into batches of 100 to be processed by the cross_validate_many script

## Merging standard results for each perspective
* Put all your dataset_algo_datetime.json files into one directory (without modifying `standards_for_uid_sets.py`, it should be `misc_standards/`)
* `python standards_for_uids_sets.py --join`
* You will get giant MERGED_ json file

## Processing results (Computing the additive and percent changes in algorithm performance)
`python process_all.py`

## 

