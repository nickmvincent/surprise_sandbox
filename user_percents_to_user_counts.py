"""
ML-1M has 6040 users.
ml-20m has 138493 users.

This very simple script takes a set of percentages (e.g. 0.1%) and outputs how many users that corresponds to.
"""
NUM_SAMPLES = 20
INDICES = '1,20'

def main():
    dataset = 'ml-20m'
    if dataset == 'ml-20m':
        num_users = 138493
    else:
        num_users = 6040

    # percents = [
    #     0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50
    # ]
    # [1.0, 3.0, 6.0, 30.0, 60.0, 302.0, 604.0, 3020.0]

    # percents = [
    #     0.01, 0.05, 0.25, 1.25, 6.25, 31.25,
    # ]
    # percents = [
    #     10, 20, 30, 40, 50, 60, 70, 80, 90, 99
    # ]
    #604.0, 1208.0, 1812.0, 2416.0, 3020.0, 3624.0, 4228.0, 4832.0, 5436.0, 5980.0


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
    jobs = []
    aws_jobs = []
    for user_count in user_counts:
        job = "python sandbox.py --grouping sample --sample_sizes {} --num_samples {} --indices {} --dataset {}".format(
            int(user_count), int(NUM_SAMPLES), INDICES, dataset
        )
        aws_job = "python3 sandbox.py --grouping sample --sample_sizes {} --num_samples {} --indices {} --dataset {} --send_to_out".format(
            int(user_count), int(NUM_SAMPLES), INDICES, dataset
        )
        jobs.append(job)
        aws_jobs.append(aws_job)
    
    with open("bash_scripts/{}_autogen_jobs.sh".format(dataset), "w") as outfile:
        outfile.write('\n'.join(jobs))

    with open("bash_scripts/{}_autogen_jobs.sh".format(dataset), "w", newline='\n') as outfile:
        outfile.write('\n'.join(aws_jobs))


main()