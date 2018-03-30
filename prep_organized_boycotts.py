"""
A script to identify all the users for a given organized boycott


ml-1m readme: http://files.grouplens.org/datasets/movielens/ml-1m-README.txt

.info() for the users_df
Data columns (total 5 columns):
user_id       6040 non-null int64
gender        6040 non-null object
age           6040 non-null int64
occupation    6040 non-null int64
zip_code      6040 non-null object
"""
import argparse
from collections import defaultdict
import json
from utils import get_dfs, GENRES

import pandas as pd

DIRECTORY = 'boycott_files'

def group_by_gender(users_df):
    """Return all men and all women"""
    return [
        {'df': users_df[users_df.gender == 'M'],
            'name': 'male users excluded'},
        {'df': users_df[users_df.gender == 'F'],
            'name': 'female users excluded'},
    ]


def group_by_age(users_df):
    """
    Return an entry for each age bin in ml-1m
    http://files.grouplens.org/datasets/movielens/ml-1m-README.txt
    """
    ret = []
    age_ranges = {
        1: "Under 18",
        18: "18-24",
        25: "25-34",
        35: "35-44",
        45: "45-49",
        50: "50-55",
        56: "56+",
    }
    for age_key, age_range in age_ranges.items():
        ret.append({
            'df': users_df[users_df.age == age_key],
            'name': age_range + ' excluded'
        })
    return ret


def group_by_occupation(users_df):
    """
    Group by occupation.
    Almost the same code as age
    http://files.grouplens.org/datasets/movielens/ml-1m-README.txt
    """
    ret = []
    num_to_occ = {
        0:  "other",
        1:  "academic/educator",
        2:  "artist",
        3:  "clerical/admin",
        4:  "college/grad student",
        5:  "customer service",
        6:  "doctor/health care",
        7:  "executive/managerial",
        8:  "farmer",
        9:  "homemaker",
        10:  "K-12 student",
        11:  "lawyer",
        12:  "programmer",
        13:  "retired",
        14:  "sales/marketing",
        15:  "scientist",
        16:  "self-employed",
        17:  "technician/engineer",
        18:  "tradesman/craftsman",
        19:  "unemployed",
        20:  "writer",
    }
    for occ_key, occ in num_to_occ.items():
        ret.append({
            'df': users_df[users_df.occupation == occ_key],
            'name': occ + ' excluded'
        })
    return ret


def group_by_genre(ratings_df, movies_df):
    """
    Group by users who like a particular genre

    How do we define a user who likes a genre?

    At least 10 ratings per genre
    Average score of 4 or higher
    * Action
	* Adventure
	* Animation
	* Children's
	* Comedy
	* Crime
	* Documentary
	* Drama
	* Fantasy
	* Film-Noir
	* Horror
	* Musical
	* Mystery
	* Romance
	* Sci-Fi
	* Thriller
	* War
	* Western
    """
    ret = []
    try:
        with open(DIRECTORY + '/users_to_genre_ratings.json', 'r') as fileobj:
            users_to_genre_ratings = json.load(fileobj)
    except FileNotFoundError:
        ratings_df = ratings_df.merge(movies_df, on='movie_id')
        print(ratings_df.head())
        users_to_genre_ratings = defaultdict(lambda: defaultdict(list))
        for i_rating, rating_row in ratings_df.iterrows():
            for genre in rating_row.genres.split('|'):
                users_to_genre_ratings[rating_row.user_id][genre].append(rating_row.rating)
        with open(DIRECTORY + '/users_to_genre_ratings.json', 'w') as fileobj:
            json.dump(users_to_genre_ratings, fileobj)

    return ret


def group_by_power(users_df):
    """
    Group power users and non-power users, based on number of contributions
    """




def group_by_state(users_df):
    """Return users from each state
    source of zip info: http://download.geonames.org/export/zip/

    from the readme: 
    country code      : iso country code, 2 characters
    postal code       : varchar(20)
    place name        : varchar(180)
    admin name1       : 1. order subdivision (state) varchar(100)
    admin code1       : 1. order subdivision (state) varchar(20)
    admin name2       : 2. order subdivision (county/province) varchar(100)
    admin code2       : 2. order subdivision (county/province) varchar(20)
    admin name3       : 3. order subdivision (community) varchar(100)
    admin code3       : 3. order subdivision (community) varchar(20)
    latitude          : estimated latitude (wgs84)
    longitude         : estimated longitude (wgs84)
    accuracy          : accuracy of lat/lng from 1=estimated to 6=centroid
    """
    ret = []
    errs = 0    
    filename = 'places_to_zips.json'
    try:
        with open(DIRECTORY + '/' + filename, 'r') as fileobj:
            places_to_zips = json.load(fileobj)
    except FileNotFoundError:
        places_to_zips = defaultdict(list)
        zip_code_df = pd.read_csv('zip_code_data/zipcodes.txt', sep='\t', encoding='utf8', header=None, usecols=[0,1,4], names=['country', 'zip_code', 'state_abbrev'], dtype=str)
        us_codes = pd.read_csv('zip_code_data/uszipcodes.txt', sep='\t', encoding='utf8', header=None, usecols=[0,1,4], names=['country', 'zip_code', 'state_abbrev'], dtype=str)
        users_df.zip_code = users_df.zip_code.astype('str')
        for zip_code in list(set(users_df.zip_code)):
            # first try to get a US zip code
            zip_row = us_codes[us_codes.zip_code == zip_code]
            if zip_row.empty:
                # next check if its an international zip code
                zip_row = zip_code_df[zip_code_df.zip_code == zip_code]
                if zip_row.empty:
                    if '-' in zip_code:
                        trunc_zip = zip_code[:5]
                        zip_row = us_codes[us_codes.zip_code == trunc_zip]
                        if zip_row.empty:
                            errs += 1
                            continue
                    else:
                        errs +=1 
                        continue
            country_abbrev = zip_row.country.iloc[0]
            state_abbrev = zip_row.state_abbrev.iloc[0]
            if country_abbrev == 'US':
                place = '{}_{}'.format(country_abbrev, state_abbrev)
            else:
                place = country_abbrev
            places_to_zips[place].append(zip_code)
        with open(DIRECTORY + '/' + filename, 'w') as fileobj:
            json.dump(places_to_zips, fileobj)
    place_to_df = {}
    for state, zips in places_to_zips.items():
        place_to_df[state] = users_df[users_df.zip_code.isin(zips)]
    for state, df in place_to_df.items():
        ret.append({
            'df': df,
            'name': 'users from state {} exlcuded'.format(state)
        })
    print(errs)
    for key, val in places_to_zips.items():
        print(key, len(val))
    return ret


def parse():
    """
    Parse args and handles list splitting
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--grouping')
    parser.add_argument('--dataset', default='ml-1m')
    args = parser.parse_args()
    dfs = get_dfs(args.dataset)
    if args.grouping == 'genre':
        print(dfs['movies'].head())
        group_by_genre(dfs['ratings'], dfs['movies'])
    else:
        grouping_to_func = {
            'gender': group_by_gender,
            'age': group_by_age,
            'state': group_by_state,
            'power': group_by_power,
        }
        grouping_to_func[args.grouping](dfs['users'])


if __name__ == '__main__':
    parse()