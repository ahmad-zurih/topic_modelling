import argparse
from calendar import monthrange
import datetime
import pickle
import random


random.seed(42)


def generate_random_minutes_for_month(month: int, year: int, n: int) -> datetime.date:
    # chosen_dates = []
    # start_date = datetime.datetime(year, month, 1, 0, 0, 0, 0)
    # end_day = monthrange(year, month)[1]
    # end_date = datetime.datetime(year, month, end_day, 0, 0, 0, 0)
    # time_between_dates = end_date - start_date
    # days_between_dates = time_between_dates.days
    # while len(chosen_dates) < n:
    #     random_number_of_days = random.randrange(days_between_dates)
    #     random_date = start_date + datetime.timedelta(days=random_number_of_days)
    #     if random_date not in chosen_dates:
    #         chosen_dates.append(random_date)
    chosen_minutes = set()
    while len(chosen_minutes) < n:
        end_day = monthrange(year, month)[1]
        random_day = random.randint(1, end_day)
        random_hour = random.randint(0, 23)
        # random_minute = random.randint(0, 59)
        chosen_min = datetime.datetime(year, month, random_day, random_hour, 0, 0, 0)
        chosen_minutes.add(chosen_min)
    return chosen_minutes


def main(args: argparse.Namespace) -> None:
    windows = {}  # {<month>: list of date-time-objects}
    for month in range(1, 13):
        chosen_windows = generate_random_minutes_for_month(month=month, year=args.year, n=args.num_time_windows_per_month)
        windows[month] = chosen_windows
        print(f'month: {month}, chosen time windows: {windows}')
    with open('results/chosen_time_windows.pickle', 'wb') as fout:
        pickle.dump(windows, fout)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_time_windows_per_month', type=int, help='Number of time windows per month to sample.')
    parser.add_argument('-y', '--year', type=int, help='The year.')
    cmd_args = parser.parse_args()
    main(cmd_args)
