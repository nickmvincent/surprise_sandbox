
python sandbox.py --grouping sample --sample_sizes 100 --num_samples 1 --dataset ml-1m --indices 0,0 > test_baselines100.txt
python sandbox.py --grouping sample --sample_sizes 1000 --num_samples 1 --dataset ml-1m --indices 0,0 > test_baselines1000.txt
python sandbox.py --grouping sample --sample_sizes 2000 --num_samples 1 --dataset ml-1m --indices 0,0 > test_baselines2000.txt
python sandbox.py --grouping sample --sample_sizes 3000 --num_samples 1 --dataset ml-1m --indices 0,0 > test_baselines3000.txt
python sandbox.py --grouping sample --sample_sizes 4000 --num_samples 1 --dataset ml-1m --indices 0,0 > test_baselines4000.txt





