"""
ML-1M has 6040 users.
ml-20m has 138493 users.

This very simple script takes a set of percentages (e.g. 0.1%) and outputs how many users that corresponds to.
"""
import os

def main():
    dataset = 'ml-1m'
    if dataset == 'ml-20m':
        num_users = 138493
        batchsize = 25
        batches = 2
        # why batches of twenty?
        # for ml-20m each dataset takes about 4GB and must be copied 
        # with imperfect (automatic) garbage collection we see 20 samples taking up to 95 GB
        # verify this please

    elif dataset == 'ml-1m':
        num_users = 6040
        batchsize = 10
        batches = 5

    configs = []
    for i in range(batches):
        start = 1 + (i * batchsize)
        end = batchsize + (i * batchsize)
        configs.append(
            (batchsize, '{},{}'.format(start, end)), # num_samples, indices
        )

    percents = [
        0.01, 0.05,
        0.1, 0.5,
        1, 5,
        10, 20, 30, 40, 50, 60, 70, 80, 90,
        99,
    ]

    user_counts = []
    for percent in percents:
        fraction = percent / 100
        user_count = round(num_users * fraction, 0)
        user_counts.append(user_count)

    user_counts = sorted(list(set(user_counts)))

    print(len(user_counts))
    print(user_counts)
    for num_samples, indices in configs:
        jobs = []
        aws_jobs = []
        grouped_jobs = []
        for user_count in user_counts:
            job = "python sandbox.py --grouping sample --sample_sizes {} --num_samples {} --indices {} --dataset {}".format(
                int(user_count), int(num_samples), indices, dataset
            )
            aws_job = "python3 sandbox.py --grouping sample --sample_sizes {} --num_samples {} --indices {} --dataset {} --send_to_out --save_path False".format(
                int(user_count), int(num_samples), indices, dataset
            )
            jobs.append(job)
            aws_jobs.append(aws_job)

        if dataset == 'ml-1m':
            groupings = [
                'gender', 'state', 'power', 'age', 'occupation', 'genre'
            ]
            for grouping in groupings:
                job = "python sandbox.py --grouping {} --num_samples {} --userfrac 0.5 --ratingfrac 1.0 --indices {} --dataset {}".format(
                    grouping, int(num_samples), indices, dataset
                )
                grouped_jobs.append(job)
            
        
        with open("bash_scripts/{}_autogen_jobs_{}.sh".format(dataset, indices), "w", newline='\n') as outfile:
            outfile.write('\n'.join(jobs))

        # s3_dir = 's3/{}_autogen_aws_{}'.format(dataset, indices)
        # if not os.path.exists(s3_dir):
        #     os.makedirs(s3_dir)
        # with open(s3_dir + '/jobs.txt', "w", newline='\n') as outfile:
        #     outfile.write('\n'.join(aws_jobs))

        # grouped_s3_dir = 's3/{}_autogen_aws_{}_grouped'.format(dataset, indices, grouping)
        # if not os.path.exists(grouped_s3_dir):
        #     os.makedirs(grouped_s3_dir) 
        # with open(grouped_s3_dir + '/jobs.txt', "w", newline='\n') as outfile:
        #     outfile.write('\n'.join(grouped_jobs))

        with open('bash_scripts/{}_autogen_jobs_{}_grouped.sh'.format(dataset, indices), 'w', newline='\n') as outfile:
            outfile.write('\n'.join(grouped_jobs))



main()
