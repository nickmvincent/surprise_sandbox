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
