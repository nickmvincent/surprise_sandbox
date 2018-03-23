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
1 per users = 9000 experiments
2, 4, 8, 16, 32, 64 = 600 experiments


demographics:
gender (2); each age group bin (10ish); each 1st digit zip (9ish); each genre (19ish);

(9000 + 600 + 40) * 28.6 / 36000