"""
ML-1M has 6040 users.

This very simple script takes a set of percentages (e.g. 0.1%) and outputs how many users that corresponds to.
"""

def main():
    num_users = 6040

    percents = [
        0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50
    ]
    # percents = [
    #     0.01, 0.05, 0.25, 1.25, 6.25, 31.25,
    # ]

    user_counts = []
    for percent in percents:
        fraction = percent / 100
        user_count = round(num_users * fraction, 0)
        user_counts.append(user_count)

    print(user_counts)

main()