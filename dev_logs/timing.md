# Timing

## 3/27
### ml-1m
`python sandbox.py --dataset ml-1m --grouping sample --sample_sizes 10 --num_samples 60`

Full runtime was: 3695.9246697425842 for 60 runs

3696 / 60
61.6 sec/run

Full runtime was: 2546.780566215515 for 54 runs
47

### ml-100k
`python sandbox.py --dataset ml-100k --grouping individual_users`
Full runtime was: 2039.7360241413116 for 1 runs


## 3/28
python sandbox.py --grouping sample --num_samples 100 --sample_sizes 2,4,8
Full runtime was: 13894.118345737457 for 300 runs


## 4/3

Full runtime was: 28615.726298093796 for 400 runs
400 total train/tests will be run because you chose 4 sample_sizes and number of samples of 100
71.54 seconds per run


## 4/6
AWS m5.large
Full runtime was: 163.09114909172058 for 1 runs
incudes download time.

AWS m5.12xlarge
Full runtime was: 2085.207819700241 for 20 runs
104 seconds per run

## 4/7/
AWS m5.12xlarge
Full runtime was: 4740.389183282852 for 100 runs
1.32 hours
47.4 seconds per run

Full runtime was: 6950.251708745956 for 150 runs

# 4/10
11904.232759714127 for 250 runs

# 4/13

Preliminary timing analysis of refactored standards calcs.

1.4 second per data split. (edit: 0.3 now.)

So data seconds of data splitting = 1.4 * num_rows * num_folds
num_rows = num_samples * num_algos

Ex: 300 samples * 2 algos * 5 folds * 1.4 = 
70 minutes

102 rows and 5 folds = 989 seconds.
80 seconds per split??? Is it all from the assert..
50 item eval batch takes 100 seconds. Doesn't appear to be threading properly.

Each of 5 fits take about 45 seconds too. Plus however much is needed for evals (which should be parallelized)

273.3401048183441 to eval a batch of 50?
about 5 seconds per eval... and this is SVD... so that seems reasonable?


# updated it
1861 per crossfold per 100 rows
(1 algo only!!!)

476.9139552116394 per crossfold w/ 100 rows
(1 algo)


725.6957314014435 seconds since last tic (doing crossfolds)
180 rows
1 algo

# 7/6/2018 running ML-20m
Full runtime was: 7067.178636074066 for 1 runs


# 7/8
`python sandbox.py --grouping sample --sample_sizes 14 --num_samples 5 --indices 1,5 --dataset ml-20m`
Full runtime was: 9482.831318616867 for 5 runs

`python sandbox.py --grouping sample --sample_sizes 69 --num_samples 10 --indices 1,10 --dataset ml-20m`
Full runtime was: 10857.623568058014 for 10 runs


# 7/11
`python sandbox.py --grouping sample --sample_sizes 138 --num_samples 5 --indices 1,5 --dataset ml-20m`
Full runtime was: 9410.30608868599 for 5 experimental iterations


`python sandbox.py --grouping sample --sample_sizes 692 --num_samples 10 --indices 1,10 --dataset ml-20m`
Full runtime was: 11257.717018604279 for 10 experimental iterations


# 7/14
27699
Full runtime was: 34552.31586265564 for 10 experimental iterations

# 7/16
dataset-ml-20m_type-sample_users_userfrac-1.0_ratingfrac-1.0_sample_size-55397_num_samples-10_indices-1-to-10.csv
Full runtime was: 115453.24959731102 for 10 experimental iterations

14
Full runtime was: 10423.584015130997 for 10 experimental iterations

