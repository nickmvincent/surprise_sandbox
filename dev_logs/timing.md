# Timing

## 3/21/2018 - ml-100k test
python sandbox.py --dataset ml-100k --grouping individual_users --num_users_to_stop_at 100

Test parameters:
ml-100k; 4 algorithms (SVD and KNN w/ 3 different similarities); 100 individual users
In total, this is 400 runs
Took 691.9 seconds (10.3 minutes)
This means each ml-100k run took ~1.7 seconds.

## 3/21/2018 - ml-1m test
python sandbox.py --dataset ml-1m --grouping individual_users --num_users_to_stop_at 100

Test parameters:
ml-1m; 2 algorithms (SVD and KNN item w/ msd similarity); 100 individual users
In total, this is 198 runs
5666.7 seconds (94.5 minutes)
28.6 seconds per run

### Tried once more...
5736.51 seconds
Definitely confirmed that 56 cpus were in use simultaneously... so what is going on?
One strange observation is that the fit time is different for different metrics...
Explanation:  the model is re-fitting for each evaluation group! So this 3x slower than it should be...

At this rate:
1 per users = 6000 experiments
2, 4, 8, 16, 32, 64 = 600 experiments


demographics:
gender (2); each age group bin (10ish); each 1st digit zip (9ish); each genre (19ish);

(6000 + 600 + 40) * 28.6 / 36000


## 3/23/2018 (after large code improvements)
python sandbox.py --dataset ml-100k --grouping individual_users --num_users_to_stop_at 100
SVD and KNNBaseline
Full runtime was: 172.3
172.3 sec / 198 runs = 0.87 sec/run. 2x improvement from before.

python sandbox.py --dataset ml-1m --grouping individual_users --num_users_to_stop_at 101
Full runtime was: 3753
3753 / 200 = 18 sec/run
Included excessive print statements and other processes running, should try this again.

### Aftering parallelization each algorithm separately
Full runtime was: 3656.2
18.28 sec/run

Analyzing this runtime:
KNN takes 26 + 12 seconds
SVD takes about 100 + 12 seconds

Together, average time is 70 seconds per run.

### After using memmap
Full runtime was: 3646


Each cross fold procedure takes:
596.9851765632629

So this means each run ACTUALLY takes 670 seconds (about 11.2 minutes)
With this in mind, the theoretical max rate should be about
670 / 56 = 11.2 second/run. Which is close to our actual speed of 18 sec/run


## 3/24
python sandbox.py --dataset ml-1m --grouping individual_users --num_users_to_stop_at 60
Full runtime was: 1195.000543832779 for 1 runs

1195 / (59*2)

python sandbox.py --dataset ml-1m --grouping individual_users --num_users_to_stop_at 101
Full runtime was: 1513.766033411026 for 1 runs

1514 / 200
7.57


## 3/25
Full runtime was: 61179.8658721447 for 1 runs
61180 / 12000
5.09


