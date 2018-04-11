"""
ML-1M has 6040 users.

This very simple script takes a set of percentages (e.g. 0.1%) and outputs how many users that corresponds to.
"""

def main():
    num_users = 6040

    # percents = [
    #     0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50
    # ]
    # percents = [
    #     0.01, 0.05, 0.25, 1.25, 6.25, 31.25,
    # ]
    percents = [
        10, 20, 30, 40, 50, 60, 70, 80, 90, 99
    ]
    #604.0, 1208.0, 1812.0, 2416.0, 3020.0, 3624.0, 4228.0, 4832.0, 5436.0, 5980.0

    user_counts = []
    for percent in percents:
        fraction = percent / 100
        user_count = round(num_users * fraction, 0)
        user_counts.append(user_count)

    print(user_counts)

main()