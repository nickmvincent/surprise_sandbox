#%% [markdown]
# # Data Strikes (and Boycotts): Simulated Campaign Results
# 
# ## What's in this notebook?
# This notebook includes code for exploring the results of simulated boycotts. It also generates plots (for papers/presentations). This notebook covers 3 broad categories of experiments
# 
# 1. MovieLens 1M random boycotts
# 2. MovieLens 1M homogeneous boycotts (e.g. fans of Comedy movies boycott together, people of a single age group boycott together)
# 3. MovieLens 20M random boycotts
# 
# It uses code from p_b_curve.py for plotting
# It preloads constants from viz_constants (# users, # ratings, # hits). Can see how those are computed in ratings_hists notebook.
# 

#%%
import viz_constants 
import importlib

# reload constants from file
# see https://support.enthought.com/hc/en-us/articles/204469240-Jupyter-IPython-After-editing-a-module-changes-are-not-effective-without-kernel-restart
importlib.reload(viz_constants)
(
    num_users, num_ratings, num_hits
) = (
    viz_constants.num_users, viz_constants.num_ratings, viz_constants.num_hits
)
num_users, num_ratings, num_hits


#%%
from collections import defaultdict, OrderedDict
import json
from pprint import pprint

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
sns.set_color_codes("dark")

import scipy
from scipy import stats
from scipy.interpolate import interp1d

# http://files.grouplens.org/datasets/movielens/ml-20m-README.txt

for dataset in ['ml-20m', 'ml-1m']:
    ratio = num_ratings[dataset] / num_users[dataset]
    print(dataset, 'ratio of ratings to users:', ratio)


#%%
print('Pandas:', pd.__version__, 'Seaborn:', sns.__version__, 'numpy:', np.__version__, 'scipy:', scipy.__version__)

#%% [markdown]
# First things first: let's load our master data files. Each is a single csv that was created by the `process_all.py` script.

#%%
df_1m = pd.read_csv('ml-1m_collected/all_results.csv', dtype={'indices': str})
df_1m['dataset'] = 'ml-1m'
assert len(df_1m) == len(df_1m[df_1m.ratingfrac.notna()])


#%%
df_20m = pd.read_csv('ml-20m_collected/all_results.csv', dtype={'indices': str})
df_20m['dataset'] = 'ml-20m'
print(len(df_20m), len(df_20m[df_20m.ratingfrac.notna()]))
      
assert len(df_20m) == len(df_20m[df_20m.ratingfrac.notna()])

#%% [markdown]
# Now we apply transformation that will affect the entire dataframe.
# 1. Calculate the number of users included in each experiment.
# 2. Calculate the number of ratings missing per experiment (so we can estimate how much we'd "expect" performance to decrease)
# 3. Calculate the number of ratings missing per experiment, rounded to the nearest 10^5
# 
# 4. Computed surfaced hits
# 
# Average # of surfaced hits for some group is equal to the totalhits for that group divided by the total # of possible hits averaged across folds
# 
# Because we use 5-fold validation, the total # of possible hits averaged across folds is total # of possible hits in the dataset divided by 5.

#%%
for df, dataset in [
    (df_1m, 'ml-1m',),
    (df_20m, 'ml-20m',),
]:
    filename_ratingcv_standards = 'standard_results/{}_ratingcv_standards_for_{}.json'.format(
        dataset, 'GlobalMean')
    with open(filename_ratingcv_standards, 'r') as f:
        global_mean_res = json.load(f)
        
    filename_ratingcv_standards = 'standard_results/{}_ratingcv_standards_for_{}.json'.format(
        dataset, 'SVD')
    with open(filename_ratingcv_standards, 'r') as f:
        svd_res = json.load(f)
    
    df['num_users_boycotting'] = [(num_users[dataset] - int(x)) / num_users[dataset] for x in df.num_users]
    df['num_ratings_missing'] = [num_ratings[dataset] - int(x) for x in df.num_ratings]
    df['nrm_rounded'] = [round(x, -5) for x in df.num_ratings_missing]
    
    new_metric = 'surfaced-hits'

    groups = ['all', 'non-boycott', 'like-boycott', 'boycott']
    if dataset == 'ml-20m':
        groups = ['all', 'non-boycott', 'boycott'] # there are no like-boycotters in 20m
    for group in groups:
        # need to compute the total number of possible hits in our tests
        # can take the average number of hits for SVD
        #denom = (svd_res['hits'] / svd_res['normhits'] * num_users[dataset])
        denom = num_hits[dataset] / 5
        
        
        # Want to see the # hits per run, # users, and # ideal hits?
        print(svd_res['totalhits'], num_users[dataset], denom)

        # just divide totalhits by # ideal hits (total # hits / # folds) 
        for a in [
            '',
            'increase_',
            'standards_',
        ]:
            key = a + new_metric + '_' + group
            print(key)
            df[key] = [x / denom for x in df[a + 'totalhits_' + group]]

    # handle 'not-like-boycott' group
    other = 'totalhits'

    if dataset == 'ml-1m': # there is no not_like-boycott in 20m
        for x in [
            other,
            'standards_' + other,
            'increase_' + other,
            new_metric,
            'standards_' + new_metric,
            'increase_' + new_metric,
        ]:
            # the total hits of not-like-boycotters and like-boycotters add up to total hits of all non boycotters
            # the same is true for surfaced hits (surfaced hits is linearly proportional to total hits)
            df[x + '_not-like-boycott'] = df[x + '_non-boycott'] - df[x + '_like-boycott']

        df['percent_increase_' + other + '_not-like-boycott'] = 100 * (
            df['increase_' + other + '_not-like-boycott']
        ) / df['standards_' + other + '_not-like-boycott']

    for group in groups:
        df['percent_increase_' + new_metric + '_' + group] = 100 * (
            df['increase_' + new_metric + '_' + group]
        ) / df['standards_' + new_metric + '_' + group]


#%%
df_20m['percent_increase_surfaced-hits_all'].mean()

#%% [markdown]
# # EDIT ME
# Define all the lists that can be used throughout. By editing this cell we can easily modify the full notebook.

#%%
id_vars = ['name','algo_name', 'indices', 'ratingfrac', 'userfrac', 'num_ratings', 'num_users', 'num_users_boycotting']
#id_vars = [x for x in id_vars if x in list(df.columns.values)]
metrics = [
    'rmse',
    'ndcg10',
    'ndcg5',
    'tailndcg10',
    'ndcgfull',
    'prec10t4',
    'rec10t4',
    'prec5t4',
    'rec5t4',
    'tailprec10t4',
    'tailrec10t4',
    #'prec5t4',
    #'tailprec5t4',
    'hits',
    'totalhits',
    'normhits',
]

organized_experiments = [
    'gender', 'state', 'power', 'genre', 'occupation', 'age'
]

# the algorithms we investigate (see old results for KNN experiments)
algo_names = [
    #'KNNBaseline_item_msd',
    'SVD'
]
# the standards we compare against
standard_algo_names = [
    'SVD', 
    'GlobalMean', 
    #'GuessThree',
    'MovieMean', 
    'KNNBaseline_item_msd',
    'KNNBasic_user_msd',
    'KNNBasic_item_msd',
]

#%% [markdown]
# ## Load and Organize Standard Results
# For any of our visualizations to be meaningful, we need standard results to compare the boycott results to. These standards let us evaluate "what is the effect of the boycott compared to other algorithms".
# 
# In particular, we compare the boycott performance to performance when using very non-personalized algorithms (MovieMean) and simpler personalized algorithms (user-based KNN, item-based KNN).
# 
# Below, we load these results from json files into Python data structures so we can use below when generating plots.

#%%
dataset_to_algo_to_metric_to_altalgo = {}
ds2standards = {}

one_mill_svd = {}
for dataset in [
    'ml-1m',
    'ml-20m',
]:
    standard_results = {}
    algo2metric2altalgo = defaultdict(lambda: defaultdict(dict))

    for algo_name in standard_algo_names:
        try:
            filename_ratingcv_standards = 'standard_results/{}_ratingcv_standards_for_{}.json'.format(
                dataset, algo_name)
            with open(filename_ratingcv_standards, 'r') as f:
                standard_results[algo_name] = json.load(f)
                if algo_name == 'SVD' and dataset == 'ml-1m':
                    one_mill_svd = standard_results[algo_name]
        except FileNotFoundError:
            print('File not found: {}'.format(filename_ratingcv_standards))
    for main_algo_name in algo_names:
        for metric in metrics:
            goodval = standard_results[main_algo_name].get(metric, 0)
            for st_algo_name in standard_algo_names:
                val = standard_results.get(st_algo_name, {}).get(metric)
                if val:
                    algo2metric2altalgo[main_algo_name][metric][st_algo_name] = (val - goodval) / goodval * 100 if goodval else 0
                else:
                    algo2metric2altalgo[main_algo_name][metric][st_algo_name] = 0
                if metric == 'normhits':

                    denom = num_hits[dataset] / 5
                    #denom = (standard_results['SVD']['hits'] / standard_results['SVD']['normhits'] * num_users[dataset])
                    print(st_algo_name)
                    # since user isn't computed for 20m
                    
                    if st_algo_name in standard_results:
                        goodval = standard_results[st_algo_name].get('totalhits', 0) / denom
                        standard_results[st_algo_name]['surfaced-hits'] = goodval
                        
                        algo2metric2altalgo[main_algo_name]['surfaced-hits'][st_algo_name] = algo2metric2altalgo[
                            main_algo_name]['totalhits'][
                            st_algo_name
                        ]
                        #algo2metric2altalgo[main_algo_name]['surfaced-hits'][st_algo_name] = (val - goodval) / goodval * 100 if goodval else 0
            if dataset == 'ml-20m':
                val = one_mill_svd.get(metric)
                if val:
                    algo2metric2altalgo[main_algo_name][metric]['1M_SVD'] = (val - goodval) / goodval * 100 if goodval else 0
                else:
                    algo2metric2altalgo[main_algo_name][metric]['1M_SVD'] = 0
            
            
    dataset_to_algo_to_metric_to_altalgo[dataset] = algo2metric2altalgo
    ds2standards[dataset] = standard_results
# uncomment to examine
# pprint(dataset_to_algo_to_metric_to_altalgo)
# pprint(ds2standards)


#%%
ds2standards['ml-1m']['SVD']

#%% [markdown]
# ## Reported Results: The personalization gap in units of surfaced hits for ML-1M and ML-2M

#%%
a = ds2standards['ml-1m']['SVD']['surfaced-hits'] - ds2standards['ml-1m']['MovieMean']['surfaced-hits']
b= ds2standards['ml-20m']['SVD']['surfaced-hits'] - ds2standards['ml-20m']['MovieMean']['surfaced-hits']
print('ML-1M:', a, '\nML-20M:', b)

#%% [markdown]
# ## Reported results: SVD standard surfaced-hits for ML-1M

#%%
round(ds2standards['ml-1m']['SVD']['surfaced-hits'] * 100, 1)


#%%
normalize_hits = False

for df, dataset in [
    (df_1m, 'ml-1m',),
    (df_20m, 'ml-20m',),
]:
    metric2altalgo = dataset_to_algo_to_metric_to_altalgo[dataset]['SVD']
    # will only work for SVD as its written right now.
    
    #personalization_boost_coefficient
    for pbc in [
        1.1, 1.02, 2, 4, 8,
    ]:
        for old_metric, k in [
            ('prec10t4', 10),
            #('ndcg10', 10),
            # the actual count of 4 star ratings in the whole ds
            ('hits', 1),
            ('totalhits', 1),
            ('normhits', 1)
        ]:
            for group in ['non-boycott', 'all']:
                # name the new dataframe column
                new_metric = '{}boosthits-{}'.format(pbc, old_metric)
                total_possible_hits = num_users[dataset] * k
                if old_metric == 'totalhits':
                    total_possible_hits = ds2standards[dataset]['SVD'][old_metric]
                elif old_metric == 'hits':
                    total_possible_hits = ds2standards[dataset]['SVD'][old_metric] * num_users[dataset]
                
                # boycotting users provide 0 hits. 
                # so with a full boycott, damage = total possible hits
                comparison = -1 * total_possible_hits
                #comparison = -1 * ds2standards[dataset]['SVD'][old_metric] * total_possible_hits

                key_template = 'percent_increase_{}_{}'
                new_key = key_template.format(new_metric, group)

   
                # no missing users in a strike ("All Users" perspective)
                if group == 'all':
                    frac_miss_arr = [0] * len(df.index)
                # there are missing users in a data boycott!
                else:
                    frac_miss_arr = list(df['num_users_boycotting'])

                labor_metric = 'labor-' + new_metric
                consumer_metric = 'consumer-' + new_metric

                # get the predicted diff if we switched to MovieMean
                pred_diff = metric2altalgo[old_metric]['MovieMean']
                # should get more negative as prec_val gets more negative
                #print(pred_diff)
                coefficients = np.polyfit(
                    [pred_diff, 0], # x values
                    [total_possible_hits/pbc - total_possible_hits, 0],# y values
                    1, # polynomial order, i.e. this is a line
                )

                df[key_template.format(labor_metric, group)] = [
                    ( coefficients[1] + coefficients[0] * prec_val  * (1 - frac_miss)) for (
                        prec_val, frac_miss
                    ) in zip(df[key_template.format(old_metric, group)], frac_miss_arr)
                ]
                df[key_template.format(consumer_metric, group)] = [
                    (comparison * frac_miss) for frac_miss in frac_miss_arr
                ]

                df[new_key] = [
                    (labor_power + consumer_power) for (
                        labor_power, consumer_power
                    ) in zip(
                        df[key_template.format('labor-' + new_metric, group)],
                        df[key_template.format('consumer-' + new_metric, group)],
                    )
                ]
                df[new_key.replace('percent_increase_', '')] = [
                    (labor_power + consumer_power + ds2standards[dataset]['SVD']['totalhits']) for (
                        labor_power, consumer_power
                    ) in zip(
                        df[key_template.format('labor-' + new_metric, group)],
                        df[key_template.format('consumer-' + new_metric, group)],
                    )
                ]

                if normalize_hits:
                    print('Normalizing. Will divide hit values by total possible hits.')
                    df[new_key] /= total_possible_hits
                    df[key_template.format(labor_metric, group)] /= total_possible_hits
                    df[key_template.format(consumer_metric, group)] /= total_possible_hits

                for st_al_nm in standard_algo_names:
                    old_val = metric2altalgo[old_metric][st_al_nm]
                    # convert from percent to raw precision change
                    old_val *= ds2standards[dataset]['SVD'][old_metric] / 100
                    # multiply by k and the probability of clicks b/c we want to get from prec@5 to hits
                    if not normalize_hits:
                        old_val *= total_possible_hits
                    metric2altalgo[new_metric][st_al_nm] = old_val
                    metric2altalgo[labor_metric][st_al_nm] = old_val
                    metric2altalgo[consumer_metric][st_al_nm] = old_val
            df['increase_hit-ratio@{}_non-boycott'.format(k)] = [
                boycott_val / strike_val for (boycott_val, strike_val) in zip(
                    df[key_template.format(new_metric, 'non-boycott')], df[key_template.format(new_metric, 'all')])
            ]
            df['increase_hit-ratio@{}_all'.format(k)] = [
                boycott_val / strike_val for (boycott_val, strike_val) in zip(
                    df[key_template.format(new_metric, 'non-boycott')], df[key_template.format(new_metric, 'all')])
            ]
            #print(df[['num_users_boycotting', 'increase_hits-prec5t4_non-boycott', 'increase_hits-prec5t4_all', 'increase_hit-ratio@5_non-boycott']])

#%% [markdown]
# ## Seperate out our data
# We want to separate the simuated boycotts into homogenous vs. heterogenous boycotts. We can also check that we have the # of samples and experiment indices that we expect.
# 
# # Reported Results: Number of Samples per strike/boycott group

#%%
samples_df_20m = df_20m[df_20m['type'] == 'sample_users']
print('\n===Heterogenous 20M Boycotts===\n')
print(samples_df_20m.name.value_counts())
print(samples_df_20m['indices'].value_counts())

samples_df_1m = df_1m[df_1m['type'] == 'sample_users']
print('\n===Heterogenous Boycotts===\n')
print(samples_df_1m.name.value_counts())
print(samples_df_1m['indices'].value_counts())

org_df = df_1m[df_1m['type'].isin(organized_experiments)]
print('\n===Homogenous Boycotts===\n')
print(org_df.name.value_counts())
print(org_df['indices'].value_counts())

#%% [markdown]
# ### Bonus analysis: Compare Precision-Estimated Hits to Actual Hits
# 
# Here we look at what happens if we use precision to estimate # hits (based on  linear `precision ~ hits` model)
# 

#%%
hits_df = samples_df_1m.copy()
hits_df = hits_df[~hits_df.hits_all.isna()]
hits_metric = 'ndcg10'
metric2 = 'normhits'
include = [
    'num_users_boycotting', 
]
hits_cols = [
    '{}_all'.format(hits_metric), 
    '{}_non-boycott'.format(hits_metric),
    #'{}_boycott'.format(hits_metric),
]
# for col in hits_cols:
#     hits_df[col] *= 0.76 / 14.75#num_users['ml-1m']

plot_metric2 = True
include += hits_cols
if plot_metric2:
    cols = [
        '{}_all'.format(metric2), 
        '{}_non-boycott'.format(metric2),
    ]

    include += cols
    
hits_df = hits_df[include].melt(id_vars='num_users_boycotting')
print(hits_df.head())

# for k in hits_df.num_users_boycotting:
#     print(k)
#     matches_from_hits = hits_df[hits_df.num_users_boycotting == k]
#     matches_from_orig = samples_df_1m[hits_df.num_users_boycotting == k]

sns.lmplot(
        x="num_users_boycotting", y="value", hue='variable', data=hits_df,
        fit_reg=False,
        x_estimator=np.mean, ci=99, 
    )
plt.show()


#%%
from collections import defaultdict
name2vals = defaultdict(list)

for name, group in hits_df.groupby('num_users_boycotting'):
    #print(name)
    for subname, subgroup in group.groupby('variable'):
        #print(subname, subgroup['value'].mean())
        name2vals[subname].append(subgroup['value'].mean())
        
print(name2vals)

#%% [markdown]
# ### Do two metrics correlate?

#%%
from scipy.stats import pearsonr, spearmanr

try:
    for group in [
        'all', 'non-boycott',
    ]:
        x = '{}_{}'.format(hits_metric,group)
        y = '{}_{}'.format(metric2, group)
        print(x, y)
        print(pearsonr(name2vals[x], name2vals[y]))
#         plt.plot(name2vals[x], name2vals[y])
#     plt.show()
except Exception as err:
    print(err)
    pass

#%% [markdown]
# ## Clean up the homogenous boycott "name" columns
# This is helpful because our homogenous boycott plots are going to very cluttered. We want to remove as much text as possible without making the plots confusing.

#%%
org_df.name = [
    x.replace('excluded', '')
    .replace('users from', '')
    .replace('using threshold 4', '')
    .replace('Top 10% contributors', 'power users')
    .strip()
    .lower()
    for x in list(org_df.name)
]
# can ignore the below warning, the code in this cell works


#%%
import p_b_curve
import importlib
importlib.reload(p_b_curve)
plot = p_b_curve.p_b_curve

template = 'Effect on {}, {}'

#%% [markdown]
# # Reported Results: Surfaced hits under 30% boycott and strike

#%%
df, ds = samples_df_1m, 'ml-1m'
print('Boycott:', round(df[(df.algo_name == 'SVD') & (df.name == '1812 user sample')]['surfaced-hits_non-boycott'].mean() * 100, 1))
print('Strike:', round(df[(df.algo_name == 'SVD') & (df.name == '1812 user sample')]['surfaced-hits_all'].mean() * 100, 1))

#%% [markdown]
# # Reported Results: Surfaced hits, normalized relative to MovieMean (un-personalized) under 30% boycott and strike

#%%
df, ds = samples_df_1m, 'ml-1m'
print('ML-1M')
pers = ds2standards['ml-1m']['MovieMean']['surfaced-hits'] - ds2standards['ml-1m']['SVD']['surfaced-hits']
print('Boycott:', round(df[(df.algo_name == 'SVD') & (df.name == '1812 user sample')]['increase_surfaced-hits_non-boycott'].mean() / pers * 100, 2))
print('Strike:', round(df[(df.algo_name == 'SVD') & (df.name == '1812 user sample')]['increase_surfaced-hits_all'].mean() / pers * 100, 2))

print('ML-20M')
df, ds = samples_df_20m, 'ml-20m'
pers = ds2standards['ml-20m']['MovieMean']['surfaced-hits'] - ds2standards['ml-20m']['SVD']['surfaced-hits']
print('Boycott:', round(df[(df.algo_name == 'SVD') & (df.name == '41548 user sample')]['increase_surfaced-hits_non-boycott'].mean() / pers * 100 , 2))
print('Strike:', round(df[(df.algo_name == 'SVD') & (df.name == '41548 user sample')]['increase_surfaced-hits_all'].mean() / pers * 100, 2))

#%% [markdown]
# # Worked Example at 30% strike

#%%
# Look at ml-1m
df, ds = samples_df_1m, 'ml-1m'
diff = ds2standards[ds]['MovieMean']['surfaced-hits'] - ds2standards[ds]['SVD']['surfaced-hits']
print('The difference between personalized (SVD, full data) and un-personalized (item mean, full data) is:', diff)
per_diff = (diff) / ds2standards[ds]['SVD']['surfaced-hits'] * 100
print('The percent difference between personalized and un-personalized is:', per_diff)

# Restrict df to SVD and 1812 user sample (30% of ML-1M)
filt = df[(df.algo_name == 'SVD') & (df.name == '1812 user sample')]


# Print all the #'s as percents rounded to one decimal.'
def represent(x):
    return round(x * 100, 3)

# get the raw #'s'
raw = {
    'participants': represent(filt['surfaced-hits_boycott'].mean()),
    'non-participants': represent(filt['surfaced-hits_non-boycott'].mean()),
    'all': represent(filt['surfaced-hits_all'].mean()),
}
print('raw hit numbers')
print(raw)

assert raw['participants'] + raw['non-participants'] == raw['all']

st = {
    'participants': represent(filt['standards_surfaced-hits_boycott'].mean()),
    'non-participants': represent(filt['standards_surfaced-hits_non-boycott'].mean()),
    'all': represent(filt['standards_surfaced-hits_all'].mean()),
}
print('standards (no campaign) hit numbers')
print(st)

# get the change
inc = {
    'participants': represent(filt['increase_surfaced-hits_boycott'].mean()),
    'non-participants': represent(filt['increase_surfaced-hits_non-boycott'].mean()),
    'all': represent(filt['increase_surfaced-hits_all'].mean()),
}
print('Change in hit numbers')
print(inc)

#print(inc['participants'], st['participants'], raw['participants'], round(raw['participants'] - st['participants'], 3))

norm_inc = {
    'participants': represent(filt['increase_surfaced-hits_boycott'].mean() / diff),
    'non-participants': represent(filt['increase_surfaced-hits_non-boycott'].mean() / diff),
    'all': represent(filt['increase_surfaced-hits_all'].mean() / diff),
}
print('normalized')
print(norm_inc)

# now get the *percent* change
per_inc = {
    'participants': represent(filt['percent_increase_surfaced-hits_boycott'].mean() / 100),
    'non-participants': represent(filt['percent_increase_surfaced-hits_non-boycott'].mean() / 100),
    'all': represent(filt['percent_increase_surfaced-hits_all'].mean() / 100),
}
print('Percent change in hit numbers')
print(per_inc)

norm_per_inc = {
    'participants': represent(filt['percent_increase_surfaced-hits_boycott'].mean() / per_diff),
    'non-participants': represent(filt['percent_increase_surfaced-hits_non-boycott'].mean() / per_diff),
    'all': represent(filt['percent_increase_surfaced-hits_all'].mean() / per_diff),
}
print('Normalized percent change in hit numbers')
print(norm_per_inc)


#%%
d = {
    '0) baseline': st,
    '1) SH after strike': raw,
    '2) change in SH': inc,
    '3) change in SH, normalized w.r.t un-personalized': norm_inc,
    '4) % change in SH': per_inc,
    '5) % change in SH, normalized w.r.t un-personalized': norm_per_inc,
}
ex1 = pd.DataFrame.from_dict(d, orient='index')
ex1 = ex1.rename(index=str, columns={
    'participants': 'strikers',
    'non-participants': 'non-strikers',
    'all': 'everyone'
})
ex1.to_csv('ex1.csv')
ex1


#%%
-0.46 / 23.23 * 100

#%% [markdown]
# ## Worked ML-20M example

#%%
# Look at ml-1m
df, ds = samples_df_20m, 'ml-20m'
diff = ds2standards[ds]['MovieMean']['surfaced-hits'] - ds2standards[ds]['SVD']['surfaced-hits']
print('The difference between personalized (SVD, full data) and un-personalized (item mean, full data) is:', diff)
per_diff = (diff) / ds2standards[ds]['SVD']['surfaced-hits'] * 100
print('The percent difference between personalized and un-personalized is:', per_diff)

filt = df[(df.algo_name == 'SVD') & (df.name == '41548 user sample')]


# Print all the #'s as percents rounded to one decimal.'
def represent(x):
    return round(x * 100, 2)

# get the raw #'s'
raw = {
    'participants': represent(filt['surfaced-hits_boycott'].mean()),
    'non-participants': represent(filt['surfaced-hits_non-boycott'].mean()),
    'all': represent(filt['surfaced-hits_all'].mean()),
}
print('raw hit numbers')
print(raw)

assert raw['participants'] + raw['non-participants'] == raw['all']

st = {
    'participants': represent(filt['standards_surfaced-hits_boycott'].mean()),
    'non-participants': represent(filt['standards_surfaced-hits_non-boycott'].mean()),
    'all': represent(filt['standards_surfaced-hits_all'].mean()),
}
print('standards (no campaign) hit numbers')
print(st)

# get the change
inc = {
    'participants': represent(filt['increase_surfaced-hits_boycott'].mean()),
    'non-participants': represent(filt['increase_surfaced-hits_non-boycott'].mean()),
    'all': represent(filt['increase_surfaced-hits_all'].mean()),
}
print('Change in hit numbers')
print(inc)

#print(inc['participants'], st['participants'], raw['participants'], round(raw['participants'] - st['participants'], 3))

norm_inc = {
    'participants': represent(filt['increase_surfaced-hits_boycott'].mean() / diff),
    'non-participants': represent(filt['increase_surfaced-hits_non-boycott'].mean() / diff),
    'all': represent(filt['increase_surfaced-hits_all'].mean() / diff),
}
print('normalized')
print(norm_inc)

# now get the *percent* change
per_inc = {
    'participants': represent(filt['percent_increase_surfaced-hits_boycott'].mean()),
    'non-participants': represent(filt['percent_increase_surfaced-hits_non-boycott'].mean()),
    'all': represent(filt['percent_increase_surfaced-hits_all'].mean()),
}
print('Percent change in hit numbers')
print(per_inc)

norm_per_inc = {
    'participants': represent(filt['percent_increase_surfaced-hits_boycott'].mean() / per_diff),
    'non-participants': represent(filt['percent_increase_surfaced-hits_non-boycott'].mean() / per_diff),
    'all': represent(filt['percent_increase_surfaced-hits_all'].mean() / per_diff),
}
print('Normalized percent change in hit numbers')
print(norm_per_inc)


#%%
filt['num_ratings']

#%% [markdown]
# # Reported Results: Figure 2, Left
# 
# ML-1m Surfaced Hits vs. Size

#%%
# Fig. 1, Upper Left
df, ds = samples_df_1m, 'ml-1m'

_ = plot(
    df[df.algo_name == 'SVD'], ds,
    show_interp=False,
    increase=False, percents=False, normalize=False,
    metrics=[
        'surfaced-hits',
        #'normhits',
    ],
    groups=[
        'all', 'non-boycott', 
        #'standards_non-boycott'
    ],
    legend=False, 
    ylabel='Surfaced Hits',
    title_template=template,
    height=4, aspect=1,
    line_names = ['SVD', 'MovieMean', 'GlobalMean'],
    print_vals=[0.3],
    ylim=(0, 0.8),
    algo2metric2altalgo=dataset_to_algo_to_metric_to_altalgo[ds],
    id_vars=id_vars,
    ds2standards=ds2standards
)

plt.savefig(ds + '_surfacedhits.png', bbox_inches='tight', dpi=300)
plt.show()

#%% [markdown]
# # Reported Results: Figure 2, right

#%%
df, ds = samples_df_20m, 'ml-20m'

_ = plot(
    df[df.algo_name == 'SVD'], ds,
    show_interp=False,
    increase=False, percents=False, normalize=False,
    metrics=[
        'surfaced-hits',
        #'normhits',
    ],
    groups=[
        'all', 'non-boycott', 
        #'standards_non-boycott'
    ],
    legend=True, 
    ylabel='Surfaced Hits',
    title_template=template,
    height=4, aspect=1,
    line_names = ['SVD', 'MovieMean', 'GlobalMean'], #line_names=['SVD', 'MovieMean', 'GlobalMean'],
    print_vals=False,
    ylim=(0, 0.8),
    algo2metric2altalgo=dataset_to_algo_to_metric_to_altalgo[ds],
    id_vars=id_vars,
    ds2standards=ds2standards
)
plt.savefig(ds + '_surfacedhits.png', bbox_inches='tight', dpi=300)
plt.show()

#%% [markdown]
# # Reported results: Figure 3, left

#%%
# Show Frac Ideal Hits from the Perspective of Remaining Users
df, ds = samples_df_1m, 'ml-1m'

_ = plot(
    df[df.algo_name == 'SVD'], ds,
    show_interp=False,
    increase=False, percents=False, normalize=False,
    metrics=[
        'surfaced-hits',
    ],
    groups=[
        'all', 
    ],
    legend=False, 
    ylabel="Surfaced Hits",
    title_template=template,
    height=4, aspect=1,
    line_names=[
        'SVD', 'KNNBasic_item_msd', 
        #'KNNBaseline_item_msd', 
        #'KNNBasic_user_msd', 
        'MovieMean'
    ],
    print_vals=[0.3],
    ylim=(0.71, 0.78),
    algo2metric2altalgo=dataset_to_algo_to_metric_to_altalgo[ds],
    id_vars=id_vars,
    ds2standards=ds2standards
)

plt.savefig(ds + '_surfacedhits_zoomed.png', bbox_inches='tight', dpi=300)
plt.show()


#%%
# ML-1M surfaced hits, normalized
df, ds = samples_df_1m, 'ml-1m'

_ = plot(
    df[df.algo_name == 'SVD'], ds,
    show_interp=False,
    increase=True, percents=True, normalize=True,
    metrics=[
        'surfaced-hits',
    ],
    groups=[
        'all', 
    ],
    legend=False, 
    ylabel="surfaced-hits",
    title_template='{}, {}',
    height=4, aspect=1,
    line_names=[
        'SVD', 'KNNBasic_item_msd', 
        #'KNNBaseline_item_msd', 
        #'KNNBasic_user_msd', 
        'MovieMean'
    ],
    print_vals=[0.3],
    #ylim=(0.71, 0.78),
    algo2metric2altalgo=dataset_to_algo_to_metric_to_altalgo[ds],
    id_vars=id_vars,
    ds2standards=ds2standards
)

#plt.savefig(ds + '_surfacedhits_zoomed.png', bbox_inches='tight', dpi=300)
plt.show()

#%% [markdown]
# # Reported Results: Figure 3, Right

#%%
# Show Frac Ideal Hits from the Perspective of Remaining Users
# Fig. 3
df, ds = samples_df_20m, 'ml-20m'

_ = plot(
    df[df.algo_name == 'SVD'], ds,
    show_interp=False,
    increase=False, percents=False, normalize=False,
    metrics=[
        'surfaced-hits',
    ],
    groups=[
        'all', 
    ],
    legend=False, 
    ylabel="Surfaced Hits",
    title_template=template,
    height=4, aspect=1,
    line_names=[
        'SVD', 'KNNBasic_item_msd', 
        #'KNNBaseline_item_msd', 
        #'KNNBasic_user_msd', 
        'MovieMean'
    ],
    print_vals=[0.3],
    ylim=(0.71, 0.78),
    algo2metric2altalgo=dataset_to_algo_to_metric_to_altalgo[ds],
    id_vars=id_vars,
    ds2standards=ds2standards
)

plt.savefig(ds + '_surfacedhits_zoomed.png', bbox_inches='tight', dpi=300)
plt.show()


#%%
# Show Frac Ideal Hits from the Perspective of Remaining Users
df, ds = samples_df_20m, 'ml-20m'

_ = plot(
    df[df.algo_name == 'SVD'], ds,
    show_interp=False,
    increase=True, percents=True, normalize=True,
    metrics=[
        'surfaced-hits',
    ],
    groups=[
        'all', 
    ],
    legend=False, 
    ylabel="sur Hits",
    title_template=template,
    height=4, aspect=1,
    line_names=[
        'SVD', 'KNNBasic_item_msd', 
        #'KNNBaseline_item_msd', 
        #'KNNBasic_user_msd', 
        'MovieMean'
    ],
    print_vals=[0.3],
    #ylim=(0.695, 0.731),
    algo2metric2altalgo=dataset_to_algo_to_metric_to_altalgo[ds],
    id_vars=id_vars,
    ds2standards=ds2standards
)

#plt.savefig(ds + '_usernormhits.png', bbox_inches='tight', dpi=300)
plt.show()


#%%
# Set colors for below "boost coefficient" plots

pinks = {}
pinks[2] = '#f8bbd0'
pinks[3] = '#f48fb1'
pinks[4] = '#f06292'
pinks[5] = '#ec407a'
pinks[6] = '#e91e63'
pinks[7] = '#d81b60'

purples = {}
purples[2] = '#e1bee7'
purples[3] = '#ce93d8'
purples[4] = '#ba68c8'
purples[5] = '#ab47bc'
purples[6] = '#9c27b0'
purples[7] = '#8e24aa'
#https://material.io/tools/color/#!/?view.left=0&view.right=0&primary.color=E1BEE7

#%% [markdown]
# ## Look at how total hits changes under the "boost coefficient" model

#%%
metric = 'totalhits'
_ = plot(
    samples_df_1m[samples_df_1m.algo_name == 'SVD'], 'ml-1m',
    show_interp=False,
    increase=True, percents=True, normalize=False,
    metrics=[
        '1.02boosthits-{}'.format(metric),
        '1.1boosthits-{}'.format(metric),
        '2boosthits-{}'.format(metric),
        '4boosthits-{}'.format(metric),
        '8boosthits-{}'.format(metric),
        
        'labor-1.02boosthits-{}'.format(metric),
        'labor-1.1boosthits-{}'.format(metric),
        'labor-2boosthits-{}'.format(metric),
        'labor-4boosthits-{}'.format(metric),
        'labor-8boosthits-{}'.format(metric),
        'consumer-1.1boosthits-{}'.format(metric),  
    ],
    groups=['all', 'non-boycott'],
    #hue='metric', col='group', row='algo_name',
    hue='metric', col='group', row='algo_name',
    
    legend=True, 
    ylabel='Change in Hits',
    title_template='Lost Hits During {} for {}',
    height=3, aspect=1,
    label_map={
        'labor-1.02boosthits-{}'.format(metric): 'Labor Power (1.02x)', 
        'labor-1.1boosthits-{}'.format(metric): 'Labor Power (1.1x)', 
        'labor-2boosthits-{}'.format(metric):  'Labor Power (2x)', 
        'labor-4boosthits-{}'.format(metric): 'Labor Power (4x)',
        'labor-8boosthits-{}'.format(metric): 'Labor Power (8x)',

        '1.02boosthits-{}'.format(metric): 'Labor + Consumer Power (1.02x)', 
        '1.1boosthits-{}'.format(metric): 'Labor + Consumer Power (1.1x)', 
        '2boosthits-{}'.format(metric): 'Labor + Consumer Power (2x)', 
        '4boosthits-{}'.format(metric): 'Labor + Consumer Power (4x)',
        '8boosthits-{}'.format(metric): 'Labor + Consumer Power (8x)',

        'consumer-1.1boosthits-{}'.format(metric): 'Consumer Power',
    },
    hue_order = [
        'labor-1.02boosthits-{}'.format(metric), 
        'labor-1.1boosthits-{}'.format(metric) ,
        'labor-2boosthits-{}'.format(metric), 
        'labor-4boosthits-{}'.format(metric),
        'labor-8boosthits-{}'.format(metric),

        '1.02boosthits-{}'.format(metric), 
        '1.1boosthits-{}'.format(metric), 
        '2boosthits-{}'.format(metric), 
        '4boosthits-{}'.format(metric),
        '8boosthits-{}'.format(metric),
        'consumer-1.1boosthits-{}'.format(metric),
    ],
    palette={
        'labor-1.02boosthits-{}'.format(metric): pinks[2], 
        'labor-1.1boosthits-{}'.format(metric): pinks[3], 
        'labor-2boosthits-{}'.format(metric):  pinks[4], 
        'labor-4boosthits-{}'.format(metric): pinks[5],
        'labor-8boosthits-{}'.format(metric): pinks[6],
        
        '1.02boosthits-{}'.format(metric): purples[2], 
        '1.1boosthits-{}'.format(metric): purples[3], 
        '2boosthits-{}'.format(metric):  purples[4], 
        '4boosthits-{}'.format(metric): purples[5],
        '8boosthits-{}'.format(metric): purples[6],
        
        'consumer-1.1boosthits-{}'.format(metric): '#fdd835',
    },
    plot_horiz_lines=True,
    line_names=['MaxDamage',], ylim=(-120000, 500),
    print_vals=None,
    algo2metric2altalgo=dataset_to_algo_to_metric_to_altalgo['ml-1m'],
    id_vars=id_vars,
    ds2standards=ds2standards
)
#plt.ylim(-114000, 200)
plt.savefig('ml-1m_totalhits.png', bbox_inches='tight', dpi=300)
plt.show()


#%%
metric = 'totalhits'
df = samples_df_20m
ds = 'ml-20m'
_ = plot(
    df[df.algo_name == 'SVD'], ds,
    show_interp=False,
    increase=True, percents=True, normalize=False,
    metrics=[
        '1.02boosthits-{}'.format(metric),
        '1.1boosthits-{}'.format(metric),
        '2boosthits-{}'.format(metric),
        '4boosthits-{}'.format(metric),
        '8boosthits-{}'.format(metric),
        
        'labor-1.02boosthits-{}'.format(metric),
        'labor-1.1boosthits-{}'.format(metric),
        'labor-2boosthits-{}'.format(metric),
        'labor-4boosthits-{}'.format(metric),
        'labor-8boosthits-{}'.format(metric),
        'consumer-1.1boosthits-{}'.format(metric),  
    ],
    groups=['all', 'non-boycott'],
    #hue='metric', col='group', row='algo_name',
    hue='metric', col='group', row='algo_name',
    
    legend=True, 
    ylabel='Change in Hits',
    title_template='Lost Hits During {} for {}',
    height=4, aspect=1,
    label_map={
        'labor-1.02boosthits-{}'.format(metric): 'Labor Power (1.02x)', 
        'labor-1.1boosthits-{}'.format(metric): 'Labor Power (1.1x)', 
        'labor-2boosthits-{}'.format(metric):  'Labor Power (2x)', 
        'labor-4boosthits-{}'.format(metric): 'Labor Power (4x)',
        'labor-8boosthits-{}'.format(metric): 'Labor Power (8x)',

        '1.02boosthits-{}'.format(metric): 'Labor + Consumer Power (1.02x)', 
        '1.1boosthits-{}'.format(metric): 'Labor + Consumer Power (1.1x)', 
        '2boosthits-{}'.format(metric): 'Labor + Consumer Power (2x)', 
        '4boosthits-{}'.format(metric): 'Labor + Consumer Power (4x)',
        '8boosthits-{}'.format(metric): 'Labor + Consumer Power (8x)',

        'consumer-1.1boosthits-{}'.format(metric): 'Consumer Power',
    },
    hue_order = [
        'labor-1.02boosthits-{}'.format(metric), 
        'labor-1.1boosthits-{}'.format(metric) ,
        'labor-2boosthits-{}'.format(metric), 
        'labor-4boosthits-{}'.format(metric),
        'labor-8boosthits-{}'.format(metric),

        '1.02boosthits-{}'.format(metric), 
        '1.1boosthits-{}'.format(metric), 
        '2boosthits-{}'.format(metric), 
        '4boosthits-{}'.format(metric),
        '8boosthits-{}'.format(metric),
        'consumer-1.1boosthits-{}'.format(metric),
    ],
    palette={
        'labor-1.02boosthits-{}'.format(metric): pinks[2], 
        'labor-1.1boosthits-{}'.format(metric): pinks[3], 
        'labor-2boosthits-{}'.format(metric):  pinks[4], 
        'labor-4boosthits-{}'.format(metric): pinks[5],
        'labor-8boosthits-{}'.format(metric): pinks[6],
        
        '1.02boosthits-{}'.format(metric): purples[2], 
        '1.1boosthits-{}'.format(metric): purples[3], 
        '2boosthits-{}'.format(metric):  purples[4], 
        '4boosthits-{}'.format(metric): purples[5],
        '8boosthits-{}'.format(metric): purples[6],
        
        'consumer-1.1boosthits-{}'.format(metric): '#fdd835',
    },
    plot_horiz_lines=True,
    line_names=['MaxDamage',], 
    #ylim=(-120000, 500),
    print_vals=None,
    algo2metric2altalgo=dataset_to_algo_to_metric_to_altalgo[ds],
    id_vars=id_vars,
    ds2standards=ds2standards
)
#plt.ylim(-114000, 200)
plt.savefig('ml-1m_complicatedhits.svg', bbox_inches='tight', dpi=300)
plt.show()

#%% [markdown]
# # Bonus results: All metrics

#%%
df, ds = samples_df_1m, 'ml-1m'

_ = plot(
    df[df.algo_name == 'SVD'], ds,
    show_interp=False,
    increase=False, percents=False, normalize=False,
    metrics=[
        'rmse',
        'ndcg10',
        'ndcg5',
        'tailndcg10',
        'ndcgfull',
        'prec10t4',
        'rec10t4',
        'prec5t4',
        'rec5t4',
        'tailprec10t4',
        'tailrec10t4',
        #'normhits',
    ],
    groups=[
        'all', 'non-boycott', 
        #'standards_non-boycott'
    ],
    legend=False, 
    ylabel='Metric',
    title_template=template,
    height=4, aspect=1,
    line_names = ['SVD', 'MovieMean',],
    print_vals=[],
    algo2metric2altalgo=dataset_to_algo_to_metric_to_altalgo[ds],
    id_vars=id_vars,
    ds2standards=ds2standards
)

plt.savefig(ds + '_othermetrics.png', bbox_inches='tight', dpi=300)
plt.show()


#%%
import p_b_curve
import importlib
importlib.reload(p_b_curve)
plot = p_b_curve.p_b_curve


#%%
# Show Frac Ideal Hits from the Perspective of Remaining Users
USE_PERCENTS = 0
USE_INCREASE = 1
NORMALIZE = 0


df, ds = samples_df_1m, 'ml-1m'

perinc_algo_to_metric_to_group_to = plot(
    df[df.algo_name == 'SVD'], ds,
    show_interp=False,
    increase=USE_INCREASE, percents=USE_PERCENTS, normalize=NORMALIZE,
    metrics=[
        'surfaced-hits',
        'totalhits',
    ],
    groups=[
        'non-boycott', 'all',
    ],
    legend=False, 
    ylabel="Fraction of Ideal Hits",
    title_template='{}, {}',
    height=4, aspect=1,
    line_names=[
        #'SVD', 'KNNBasic_item_msd', 
        #'KNNBaseline_item_msd', 
        #'KNNBasic_user_msd', 
        #'MovieMean'
    ],
    print_vals=[0.3],
    algo2metric2altalgo=dataset_to_algo_to_metric_to_altalgo[ds],
    id_vars=id_vars,
    ds2standards=ds2standards
)

#plt.savefig(ds + '_percentchange.png', bbox_inches='tight', dpi=300)
plt.show()

#%% [markdown]
# ## Compute the 'algo_to_metric_to_group_to' for 1m in percent change

#%%
# Show Frac Ideal Hits from the Perspective of Remaining Users
USE_PERCENTS = 1
USE_INCREASE = 1
NORMALIZE = 0


df, ds = samples_df_1m, 'ml-1m'

perinc_algo_to_metric_to_group_to = plot(
    df[df.algo_name == 'SVD'], ds,
    show_interp=False,
    increase=USE_INCREASE, percents=USE_PERCENTS, normalize=NORMALIZE,
    metrics=[
        'surfaced-hits',
        'totalhits',
    ],
    groups=[
        'non-boycott', 'all',
    ],
    legend=False, 
    ylabel="Fraction of Ideal Hits",
    title_template='{}, {}',
    height=4, aspect=1,
    line_names=[
        'SVD', 'KNNBasic_item_msd', 
        #'KNNBaseline_item_msd', 
        #'KNNBasic_user_msd', 
        'MovieMean'
    ],
    ylim=(-2.0, 0.0),
    print_vals=[0.3],
    algo2metric2altalgo=dataset_to_algo_to_metric_to_altalgo[ds],
    id_vars=id_vars,
    ds2standards=ds2standards
)

plt.savefig(ds + '_percentchange.png', bbox_inches='tight', dpi=300)
plt.show()

#%% [markdown]
# ## Compute the 'algo_to_metric_to_group_to' for 1m in normalized percent change

#%%
# Show Frac Ideal Hits from the Perspective of Remaining Users
USE_PERCENTS = 1
USE_INCREASE = 1
NORMALIZE = 1

df, ds = samples_df_1m, 'ml-1m'

algo_to_metric_to_group_to = plot(
    df[df.algo_name == 'SVD'], ds,
    show_interp=True,
    increase=USE_INCREASE, percents=USE_PERCENTS, normalize=NORMALIZE,
    metrics=[
        'ndcg10', 
        #'normhits',
        'surfaced-hits',
        #'ndcgfull',
        'totalhits'
    ],
    groups=[
        'non-boycott', 'all',
    ],
    legend=False, 
    ylabel="Fraction of Ideal Hits",
    title_template='{}, {}',
    height=4, aspect=1,
    line_names=[
        'SVD', 'KNNBasic_item_msd', 
        #'KNNBaseline_item_msd', 
        #'KNNBasic_user_msd', 
        'MovieMean'
    ],
    ylim=(-1.1, 0),
    print_vals=[0.3],
    algo2metric2altalgo=dataset_to_algo_to_metric_to_altalgo[ds],
    id_vars=id_vars,
    ds2standards=ds2standards
)

plt.savefig(ds + '_norm.png', bbox_inches='tight', dpi=300)
plt.show()


#%%
metric = 'totalhits'
metric = 'surfaced-hits'
# comparisons = ds2standards['ml-1m']
# comparisons = {
#     key: comparisons[key].get(metric, 0) for key in comparisons.keys()
# }
comparisons = dataset_to_algo_to_metric_to_altalgo['ml-1m']['SVD'][metric]
#movie_val = -1 * dataset_to_algo_to_metric_to_altalgo['ml-1m']['SVD'][metric]['MovieMean']
for group in [
    'all', 
    #'non-boycott'
]:
    found_item, found_user = False, False
    found_itembasic = False
    for x in range(0, num_ratings['ml-1m'], int(num_ratings['ml-1m'] / 10000)):
        y = perinc_algo_to_metric_to_group_to['SVD'][metric][group]['interp_ratings'](x)
        #print(x,y)
        #print(comparisons['KNNBasic_item_msd'])

#         if not found_item:
#             if y <= comparisons['KNNBaseline_item_msd']:
#                 print(group + ' found item:')
#                 print(x / num_ratings['ml-1m'])
#                 found_item = True
#         if not found_user:
#             if y <= comparisons['KNNBasic_user_msd']:
#                 print(group + ' found user')
#                 print(x / num_ratings['ml-1m'])
#                 found_user = True
        if not found_itembasic:
            if y <= comparisons['KNNBasic_item_msd']:
                print(group + ' found knnbasicitem')
                print(x / num_ratings['ml-1m'])
                found_itembasic = True


#%%
xvals = np.array(range(0, num_ratings['ml-1m'], int(num_ratings['ml-1m'] / 10000)))
arr = abs(perinc_algo_to_metric_to_group_to['SVD']['surfaced-hits']['all']['interp_ratings'](xvals) - comparisons['KNNBasic_item_msd'])
x = np.argmin(arr)
xvals[x]/1e6*100


#%%
df, ds = samples_df_1m, 'ml-1m'

inc_algo_to_metric_to_group_to = plot(
    df[df.algo_name == 'SVD'], ds,
    show_interp=False,
    increase=True, percents=False, normalize=False,
    metrics=[
        'surfaced-hits',
        #'normhits',
    ],
    groups=[
        'all', 'non-boycott', 
        #'standards_non-boycott'
    ],
    legend=False, 
    ylabel='Surfaced Hits',
    title_template=template,
    height=4, aspect=1,
    line_names = [
        #'SVD', 'MovieMean', 'GlobalMean'
    ],
    print_vals=[],
    #ylim=(0, 0.8),
    algo2metric2altalgo=dataset_to_algo_to_metric_to_altalgo[ds],
    id_vars=id_vars,
    ds2standards=ds2standards
)

#plt.savefig(ds + '_surfacedhits.png', bbox_inches='tight', dpi=300)
plt.show()


#%%
df, ds = samples_df_20m, 'ml-20m'

twentyinc_algo_to_metric_to_group_to = plot(
    df[df.algo_name == 'SVD'], ds,
    show_interp=False,
    increase=True, percents=False, normalize=False,
    metrics=[
        'surfaced-hits',
        #'normhits',
    ],
    groups=[
        'all', 'non-boycott', 
        #'standards_non-boycott'
    ],
    legend=False, 
    ylabel='Surfaced Hits',
    title_template=template,
    height=4, aspect=1,
    line_names = [
        #'SVD', 'MovieMean', 'GlobalMean'
    ],
    print_vals=[],
    #ylim=(0, 0.8),
    algo2metric2altalgo=dataset_to_algo_to_metric_to_altalgo[ds],
    id_vars=id_vars,
    ds2standards=ds2standards
)

#plt.savefig(ds + '_surfacedhits.png', bbox_inches='tight', dpi=300)
plt.show()


#%%
# Show Frac Ideal Hits from the Perspective of Remaining Users
USE_PERCENTS = 1
USE_INCREASE = 1
NORMALIZE = 1

df, ds = samples_df_20m, 'ml-20m'

# save this data structure w/ interpolation
twenty_algo_to_metric_to_group_to = plot(
    df[df.algo_name == 'SVD'], ds,
    show_interp=False,
    increase=USE_INCREASE, percents=USE_PERCENTS, normalize=NORMALIZE,
    metrics=[
        'ndcg10', 
        'surfaced-hits',

    ],
    groups=[
        'all', 'non-boycott',
    ],
    legend=False, 
    ylabel="Surfaced Hits",
    title_template=template,
    height=4, aspect=1,
    line_names=[
        'SVD', 'KNNBasic_item_msd',
        'MovieMean'
    ],
    algo2metric2altalgo=dataset_to_algo_to_metric_to_altalgo[ds],
    id_vars=id_vars,
    ds2standards=ds2standards
)

#plt.savefig(ds + '_usernormhits.png', bbox_inches='tight', dpi=300)
plt.show()

#%% [markdown]
# # Reported Results: Effects relative to non-personalized results
# ## First look at the relative change in surfaced hit for non-participants

#%%
# effects on non-participants, normalized 
d1= {}
group = 'non-boycott'
print(group)
d1['30percent_boycott_ml-1m'] = round(algo_to_metric_to_group_to['SVD']['surfaced-hits'][group]['interp_ratings'](0.3 * num_ratings['ml-1m']) * 100, 1)
d1['30percent_boycott_ml-20m'] = round(twenty_algo_to_metric_to_group_to['SVD']['surfaced-hits'][group]['interp_ratings'](0.3 * num_ratings['ml-20m']) * 100, 1)
#d1['30percent_boycott_ml-1m_ndcg10'] = round(algo_to_metric_to_group_to['SVD']['ndcg10'][group]['interp_ratings'](0.3 * num_ratings['ml-1m']) * 100, 1)
#d1['30percent_boycott_ml-20m_ndcg10'] = round(twenty_algo_to_metric_to_group_to['SVD']['ndcg10'][group]['interp_ratings'](0.3 * num_ratings['ml-20m']) * 100, 1)

pprint(d1)

print('Where does ML-20M intersect the ml-1m 30 percent line')
xvals = np.array(range(0, num_ratings['ml-20m'], int(num_ratings['ml-20m'] / 100)))
arr = abs(twenty_algo_to_metric_to_group_to['SVD']['surfaced-hits'][group]['interp_ratings'](xvals) - d1['30percent_boycott_ml-1m']/100)
x = np.argmin(arr)
print(xvals[x]/20e6*100, twenty_algo_to_metric_to_group_to['SVD']['surfaced-hits']['all']['interp_ratings'](xvals[x]))


#%%
round(algo_to_metric_to_group_to['SVD']['surfaced-hits']['non-boycott']['interp_ratings'](299826.82) * 100, 3)


#%%
d2 = {}
group = 'all'
print(group)
d2['30percent_strike_ml-1m'] = round(algo_to_metric_to_group_to['SVD']['surfaced-hits'][group]['interp_ratings'](0.3 * num_ratings['ml-1m']) * 100, 1)
d2['37.5percent_strike_ml-1m'] = round(algo_to_metric_to_group_to['SVD']['surfaced-hits'][group]['interp_ratings'](0.375 * num_ratings['ml-1m']) * 100, 1)

d2['30percent_strike_ml-20m'] = round(twenty_algo_to_metric_to_group_to['SVD']['surfaced-hits'][group]['interp_ratings'](0.3 * num_ratings['ml-20m']) * 100, 1)
pprint(d2)

#%% [markdown]
# ## Let's look at the change in surfaced hits for the perspective of non-participants (marked "non-boycott")

#%%
d3 = {}
group = 'non-boycott'
d3['30_ml-1m'] = round(inc_algo_to_metric_to_group_to['SVD']['surfaced-hits'][group]['interp_ratings'](0.3 * num_ratings['ml-1m']) * 100, 4)
d3['30_ml-20m'] = round(twentyinc_algo_to_metric_to_group_to['SVD']['surfaced-hits'][group]['interp_ratings'](0.3 * num_ratings['ml-20m']) * 100, 4)
d3['30_ml_all-20m'] = round(twentyinc_algo_to_metric_to_group_to['SVD']['surfaced-hits']['all']['interp_ratings'](0.3 * num_ratings['ml-1m']) * 100, 4)

pprint(d3)

#%% [markdown]
# # Homogenous (i.e. Targeted/Focus/shared-characteristic) Boycotts

#%%
def half_users(df):
    return df[(
        (df.userfrac == 0.5) & (df.ratingfrac == 1.0) 
        #& (df.algo_name == 'SVD')
    )]


#%%
def half_ratings(df):
    return df[(
        (df.userfrac == 1.0) & (df.ratingfrac == 0.5)
    )]


#%%
def all_users_all_ratings(df):
    return df[(
        (df.userfrac == 1.0) & (df.ratingfrac == 1.0)
    )]


#%%

#metrics += ['hits-prec5t4', 'labor-hits-prec5t4', 'consumer-hits-prec5t4']

metrics = [
    #'surfaced-hits', 
    'totalhits'
]

# warning: you probably do not want to change these (unless you're sure)
NORMALIZE = 0
USE_PERCENTS = 1
USE_INCREASE = 1

copy_org_df = org_df.copy()
if NORMALIZE:
    for metric in metrics:
        for algo_name in algo_names:
            movie_val = abs(dataset_to_algo_to_metric_to_altalgo['ml-1m']['SVD'][metric]['MovieMean'])
            for group in ['non-boycott', 'like-boycott', 'not-like-boycott']:
                col = 'percent_increase_{}_{}'.format(metric, group)

                copy_org_df.loc[
                    copy_org_df.algo_name == algo_name, col
                ] = org_df.loc[org_df.algo_name == algo_name, col] / movie_val
                
for metric in metrics:
    expec = []
    col = '{}_expected'.format(metric)
    if USE_INCREASE:
        col = 'increase_' + col
    if USE_PERCENTS:
        col = 'percent_' + col
    for i, row in copy_org_df.iterrows():
        x = row.num_ratings_missing
        expec.append(float(perinc_algo_to_metric_to_group_to[row.algo_name][metric]['non-boycott']['interp_ratings'](x)))
    kwargs = {col: expec}
    copy_org_df = copy_org_df.assign(**kwargs)

#%% [markdown]
# ## LB = like-boycott
# ### So lbless is the number cases where the like boycott performance change was less than expected

#%%
from p_b_curve import select_cols, fill_in_longform
ylabel = 'ylabel'

def plot2(
        df, metrics, groups,
        increase=False, percents=False, kind='bar', height=4, flip=False, filename='tmp.svg', save=False,
        palette=None, aspect=1
    ):
    """
    Plots the results of homogenous boycotts
    e.g. 50% of Comedy fans boycott together
    """
    print('len of df (number of experiments included)', len(df.index))
    for name in list(set(df.name)):
#        tmp = df[df.name == name].num_ratings_missing.mean()
        tmp = df[df.name == name].num_users_boycotting.mean()
        print('Num users boycotting: {}. As a percent: {}'.format(tmp, tmp*100))
#         print('name', name)
#         print('mean number of ratings missing', tmp)pl
#         print(algo_to_metric_to_group_to['SVD']['ndcg10']['non-boycott']['interp_ratings'](tmp))
    increase_cols = select_cols(list(df.columns.values), metrics, groups, increase=increase, percents=percents)
    increase_cols = [x for x in increase_cols if 'vanilla' not in x]
    longform = df[increase_cols + id_vars].melt(
        id_vars = id_vars,
        var_name='increase_type'
    )
    longform = fill_in_longform(longform)
    if flip:
        x = "name"
        y = "value"
    else:
        x = "value"
        y = "name"
    if '?' in filename:
        legend = False
    else:
        legend = True
    grid = sns.catplot(
        x=x, y=y, hue="group", data=longform,
        height=height, kind=kind, col='algo_name', row='metric',
        sharex=False,
        row_order=metrics,
        legend=legend,
        legend_out=True,
        palette=palette,
        aspect=aspect
    )
    a = grid.axes
    
    name_to_ratios = defaultdict(list)
    diffs = defaultdict(dict)
    ratios = defaultdict(dict)
    vals = defaultdict(dict)
    name_to_counts = defaultdict(lambda: defaultdict(int))
    
    for x in grid.facet_data():
        i_row, i_col, i_hue = x[0]
        metric = grid.row_names[i_row]
        algo_name = grid.col_names[i_col]
        if grid.hue_names:
            group = grid.hue_names[i_hue]        
        
        if NORMALIZE:
            val = -1
        else:
            val = abs(dataset_to_algo_to_metric_to_altalgo['ml-1m']['SVD'][metric]['MovieMean'])
        
        if flip:
            grid.axes[i_row, i_col].axhline(0, color='0.7', linestyle='-')
        else:
            grid.axes[i_row, i_col].axvline(0, color='0.7', linestyle='-')
            grid.axes[i_row, i_col].axvline(val, color='0.7', linestyle='-')
        
        cols = {}
        for col in increase_cols:
            if metric == col.split('_')[-2]:
                
                if 'expected' in col:
                    cols['expec'] = col
                if 'not-like' in col:
                    cols['not-like'] = col
                elif 'non-boycott' in col:
                    cols['nb'] = col
                elif 'like-boycott' in col:
                    cols['lb'] = col
        for name in list(set(longform.name)):
            mask = (
                (longform.metric == metric) &
                (longform.algo_name == algo_name) &
                (longform.name == name)
            )
            
            masked = longform[mask]
            try:
                arrs = {}
                print(cols)
                for key in cols:
                    arrs[key] = np.array(masked[masked.increase_type == cols[key]].value)
                
                means = {key: np.mean(arrs[key]) for key in arrs}
                print(means)
                
                ratio = means['nb'] / means['expec']
                diff = means['nb'] - means['expec']
                
                lb_diff = means['lb'] / np.mean(np.array(masked.num_users_boycotting))#- means['not-like']
                lb_ratio = means['lb'] / means['not-like']

                pval = stats.ttest_ind(arrs['expec'],arrs['nb'], equal_var=False).pvalue                
                name_to_counts[name]['total'] += 1
                
                print('{} {} {}, {}, {}'.format(metric, algo_name, name, cols['expec'], cols['nb']))
                print('Ratio: {}, pval: {}'.format(ratio, pval))
                print('LB to NB ratio: {}'.format(lb_ratio))
                print('lb_diff, mean_lb - mean_nl', lb_diff)
                print('pos = lb has more hits')
                name_to_ratios[name].append(ratio)
                diffs['nb'][name] = diff
                diffs['lb'][name] = lb_diff
                
                ratios['nb'][name] = ratio
                ratios['lb'][name] = lb_ratio
                
                vals['nb'][name] = means['nb']
                vals['lb'][name] = means['lb']
                vals['not-like'][name] = means['not-like']
                
                if pval < 0.5:
                    name_to_counts[name]['total_sig'] += 1

                if diff < 0:
                    name_to_counts[name]['total_less'] += 1
                else:
                    name_to_counts[name]['total_more'] += 1
                #print('lb info', lb_diff, lb_mean, nb_mean)
                if lb_diff < 0:
                    name_to_counts[name]['total_lbless'] += 1
            except Exception as err:
                print(err)
                print(algo_name, metric)
    total = 0
    total_sig = 0
    total_less = 0
    total_more = 0
    total_lbless = 0
    
    for name, counts in name_to_counts.items():
        total += counts['total']
        total_sig += counts['total_sig']
        total_less += counts['total_less']
        total_more += counts['total_more']
        total_lbless += counts['total_lbless']

    print('Totals:')
    print('{} lbless, {} sig, {} less, and {} more out of {} total'.format(
            total_lbless, total_sig, total_less,
            total_more, total)
        )
    if save:
        new_labels = ['Similar Users', 'Not Participating\n Users', 'Expected']
        for t, l in zip(grid._legend.texts, new_labels): t.set_text(l)
        if 'h2' in filename:
            grid.set_ylabels("")
        else:
            grid.set_ylabels(ylabel)
        grid.set_xlabels("")
        grid.set_titles("")
        plt.savefig(filename, size=(3, 3), bbox_inches='tight', dpi=300)
    return grid, (total, total_sig, total_less, total_more, total_lbless), (diffs, ratios, vals)


#%%
palette={'all': 'b', 'non-boycott': 'g', 'expected': 'y', 'like-boycott': 'c', 'not-like-boycott': 'r'}

def plot_all_three_scenarios(df, height=6):

    total, total_sig, total_less, total_more, total_lbless = 0, 0, 0, 0, 0
    
    half_users_df = half_users(df)
    if not half_users_df.empty:
        grid, (total2, total_sig2, total_less2, total_more2, total_lbless2), (diffs, ratios, vals) = plot2(
            half_users_df,
            metrics=metrics,
            increase=USE_INCREASE, percents=USE_PERCENTS,
            flip=True,
            groups=[
                'non-boycott', 'like-boycott', 
                'expected', 
                'not-like-boycott',
            ],
            height=height,
            kind='bar',
            palette=palette
        )
        grid.fig.suptitle('Half Users')
        total += total2
        total_sig += total_sig2
        total_less += total_less2
        total_more += total_more2
        total_lbless += total_lbless2
        print('total2', total2, total)
    
    return (total, total_sig, total_less, total_more, total_lbless), (diffs, ratios, vals)


#%%
copy_org_df = copy_org_df[copy_org_df.algo_name == 'SVD']


#%%
gender_boycotts = copy_org_df[copy_org_df['type'] == 'gender']
age_boycotts = copy_org_df[copy_org_df['type'] == 'age']
occupation_boycotts = copy_org_df[copy_org_df['type'] == 'occupation']
power_boycotts = copy_org_df[copy_org_df['type'] == 'power']
genre_boycotts = copy_org_df[copy_org_df['type'] == 'genre']


#%%
running_totals = {
    'total': 0,
    'total_sig': 0,
    'total_less': 0,
    'total_more': 0,
    'total_lbless': 0
}
all_diffs = {
    'lb': {},
    'nb': {}
}
all_vals = {
    'lb': {},
    'nb': {},
    'not-like': {},
}
all_ratios = {
    'lb': {},
    'nb': {}
}


#%%
df1 = half_users(gender_boycotts)
df2 = half_users(genre_boycotts)
df3 = half_users(power_boycotts)

grid, (total, total_sig, total_less, total_more, total_lbless), (diffs, ratios, vals) = plot2(
    pd.concat([df1[
        #(df1.name.str.contains('female')) & 
        (df1.algo_name == 'SVD')],
        df2[(df2.name.str.contains('documentary')) & (df2.algo_name == 'SVD')],
        df3[(df3.name.str.contains('power')) & (df3.algo_name == 'SVD')]
    
    ]),
    metrics=metrics,
    increase=USE_INCREASE, percents=USE_PERCENTS,
    groups=[
        #'non-boycott', 
        'like-boycott', 
        #'expected',
        'not-like-boycott',
    ],
    height=3.5,
    kind='bar',
    flip=True,
    filename='h1.svg', save=False,
    palette=palette,
    aspect=1.5
)

plt.show()

#%% [markdown]
# ## Male User and Female User Boycotts

#%%
(total, total_sig, total_less, total_more, total_lbless), (diffs, ratios, vals) = plot_all_three_scenarios(gender_boycotts)
running_totals['total'] += total
running_totals['total_sig'] += total_sig
running_totals['total_less'] += total_less
running_totals['total_more'] += total_more
running_totals['total_lbless'] += total_lbless
for key in all_diffs.keys():
    all_diffs[key].update(diffs[key])
    all_ratios[key].update(ratios[key])
for key in all_vals.keys():
    all_vals[key].update(vals[key])
    
print(all_vals)

plt.show()

#%% [markdown]
# # Power Boycotts
# 
# Below, the power boycotts show a very weird results. Warrants double checks.

#%%
(total, total_sig, total_less, total_more, total_lbless), (diffs, ratios, vals) = plot_all_three_scenarios(power_boycotts)
running_totals['total'] += total
running_totals['total_sig'] += total_sig
running_totals['total_less'] += total_less
running_totals['total_more'] += total_more
running_totals['total_lbless'] += total_lbless

for key in all_diffs.keys():
    all_diffs[key].update(diffs[key])
    all_ratios[key].update(ratios[key])
for key in all_vals.keys():
    all_vals[key].update(vals[key])


plt.show()

#%% [markdown]
# # Age Boycotts

#%%
(total, total_sig, total_less, total_more, total_lbless), (diffs, ratios, vals) = plot_all_three_scenarios(age_boycotts)
running_totals['total'] += total
running_totals['total_sig'] += total_sig
running_totals['total_less'] += total_less
running_totals['total_more'] += total_more
running_totals['total_lbless'] += total_lbless

for key in all_diffs.keys():
    all_diffs[key].update(diffs[key])
    all_ratios[key].update(ratios[key])
for key in all_vals.keys():
    all_vals[key].update(vals[key])

plt.show()

#%% [markdown]
# # Occupation Boycotts

#%%
(total, total_sig, total_less, total_more, total_lbless), (diffs, ratios, vals) = plot_all_three_scenarios(occupation_boycotts)
running_totals['total'] += total
running_totals['total_sig'] += total_sig
running_totals['total_less'] += total_less
running_totals['total_more'] += total_more
running_totals['total_lbless'] += total_lbless

for key in all_diffs.keys():
    all_diffs[key].update(diffs[key])
    all_ratios[key].update(ratios[key])
for key in all_vals.keys():
    all_vals[key].update(vals[key])
plt.show()

#%% [markdown]
# # Genre Boycotts

#%%
(total, total_sig, total_less, total_more, total_lbless), (diffs, ratios, vals) = plot_all_three_scenarios(genre_boycotts)
running_totals['total'] += total
running_totals['total_sig'] += total_sig
running_totals['total_less'] += total_less
running_totals['total_more'] += total_more
running_totals['total_lbless'] += total_lbless

for key in all_diffs.keys():
    all_diffs[key].update(diffs[key])
    all_ratios[key].update(ratios[key])
    
for key in all_vals.keys():
    all_vals[key].update(vals[key])

plt.show()

#%% [markdown]
# # Reported Results: Total # of groups with overall effect less than expected

#%%
running_totals


#%%
k_max = max(all_ratios['nb'], key=all_ratios['nb'].get)
print(k_max, all_ratios['nb'][k_max])

k_min = min(all_ratios['nb'], key=all_ratios['nb'].get)
print(k_min, all_ratios['nb'][k_min])


#%%
count = 0
for k, v in all_ratios['nb'].items():
    if v > 1:
        count += 1
count


#%%
from pprint import pprint
# pprint(all_diffs)
pprint(all_ratios)
# pprint(all_vals)


#%%
with open('all_ratios.json', 'w') as f:
    json.dump(all_ratios, f)
with open('all_diffs.json', 'w') as f:
    json.dump(all_diffs, f)
with open('all_vals.json', 'w') as f:
    json.dump(all_vals, f)


#%%
with open('some_distances.json', 'r') as f:
    group_to_group_to_vectype_to_distancetype_to_ = json.load(f)
    
row_dicts = []
for group, group_to_vectype_to_distancetype_to_ in group_to_group_to_vectype_to_distancetype_to_.items():
    if group == 'all':
        continue
    row_dict = {}
    row_dict['name'] = group
    for group2, vectype_to_distancetype_to_ in group_to_vectype_to_distancetype_to_.items():
        if group2 != 'all':
            continue
        for vectype, distancetype_to_ in vectype_to_distancetype_to_.items():
            for distancetype, val in distancetype_to_.items():
                row_dict['{}_{}'.format(vectype, distancetype)] = val
    row_dicts.append(row_dict)

df = pd.DataFrame(row_dicts)
df.name = [
    x.replace('excluded', '')
    .replace('users from', '')
    .replace('using threshold 4', '')
    .replace('Top 10% contributors', 'power users')
    .strip()
    .lower()
    for x in list(df.name)
]


#%%
with open('all_ratios.json', 'r') as f:
    all_ratios = json.load(f)
    
with open('all_diffs.json', 'r') as f:
    all_diffs = json.load(f)
    
with open('all_vals.json', 'r') as f:
    all_vals = json.load(f)
    
with open('group_to_num_ratings.json', 'r') as f:
    group_to_num_ratings = json.load(f)
temp_df = pd.DataFrame.from_dict(group_to_num_ratings, orient='index')
temp_df.index = [
    x.replace('excluded', '')
    .replace('users from', '')
    .replace('using threshold 4', '')
    .replace('Top 10% contributors', 'power users')
    .strip()
    .lower()
    for x in list(temp_df.index)
]
print(temp_df)
group_to_num_ratings = temp_df.to_dict()[0]


#print(group_to_num_ratings)    
lb_ratios = all_ratios['lb']
nb_ratios = all_ratios['nb']



row_dicts = []
for key, val in lb_ratios.items():
    row_dict = {
        'name': key,
        'like-boycott-ratio': val,
        'non-boycott-ratio': nb_ratios[key],
        'like-boycott-diff': all_diffs['lb'][key],
        'non-boycott-diff': all_diffs['lb'][key],
        'num_ratings': group_to_num_ratings[key],
        'like-boycott-val': all_vals['lb'][key],
        'not-like-val': all_vals['not-like'][key],
    }
    row_dicts.append(row_dict)
ratios_df = pd.DataFrame(row_dicts)

df = df.merge(right=ratios_df, on='name', how='inner')
print(df.head())


#%%
names = {
    'name': 'Name', 'num_ratings': '# Ratings', 
    'like-boycott-ratio': 'Similar User Effect Ratio',
    'like-boycott-val':  '% change surfaced hits, similar users',
    'not-like-val':  '% change surfaced hits, other users',
}

table_df = df[['name', 'num_ratings', 'like-boycott-val', 'not-like-val', 'like-boycott-ratio']][
        df.name.isin([
        'male users', 'female users', 'power users',
        'fans of drama', 'fans of horror', 
        '25-34', '56+', 'under 18',
        'lawyer', 
        #'scientist', 
        'artist',
    ])
    #df.num_ratings > 150000
].rename(index=str, columns=names)

print(table_df)

#%% [markdown]
# # Reported Results: Table 1

#%%
html = table_df.to_html(
    index=False,
    float_format='%.2f',
    columns=[
        'Name', '# Ratings', 
        names['like-boycott-val'],
        names['not-like-val'],
        'Similar User Effect Ratio'
]
)
css = """
<style>
body {column-count: 1 !important;}
table {
    width: 4.5in; height:3in;
        font-size: 9pt;
}
td {
    border: 1px solid;
}
th, td, table {
    border-left: none;
    border-right: none;
    padding-left: 10px;
}

</style>
"""
table_df.to_csv('table1.csv', index=False, float_format='%.2f', columns=[
        'Name', '# Ratings', 
        names['like-boycott-val'],
        names['not-like-val'],
        'Similar User Effect Ratio'
])

with open('table.html', 'w') as f:
    f.write('<link rel="stylesheet" href="pubcss-acm-sigchi.css">' + css + html)

#%% [markdown]
# # Reported Results: Figure 4 and Correlation
# 

#%%
# Implicit Cosine vs. LB Ratio
sns.set(style="darkgrid")
sns.set_color_codes("dark")

fig, ax = plt.subplots(1,1, figsize=(7, 3.5))
filtered = df[df.num_ratings >= 20000]
print('len, std, mean', len(filtered.num_ratings), np.std(filtered.num_ratings), np.mean(filtered.num_ratings))
#filtered = df

y = 'like-boycott-ratio'

sns.scatterplot(
    x='implicit_cosine', y=y, data=filtered,
    ax=ax, 
    #marker=".", 
    #line_kws={'alpha':.3},
    alpha=0.3
)

def label_point(x, y, val, ax, names):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        if str(point['val']) in names:
            ax.text(
                point['x'], point['y']+0.2, str(point['val']), 
            )
            ax.plot(point['x'], point['y'], 'bx')

label_point(
    filtered.implicit_cosine, filtered[y], filtered.name, plt.gca(),
    names=[
        'male users', 'female users', 'fans of film-noir', 'power users', 'under 18', 'artist',
        'fans of documentary', 'fans of horror',]
)  

plt.xlabel('Uniqueness: Group Implicit Rating Cosine Distance from Centroid')
plt.title('Similar User Effect Ratio and Uniqueness')
plt.ylabel('Similar User Effect Ratio')

ax.axhline(1, color='0.3', linestyle='--')
plt.savefig('implicitcosine_vs_lbratio.png', bbox_inches='tight', dpi=300)
plt.show()
pearson_val, pearson_p = pearsonr(filtered['implicit_cosine'], filtered[y])
print(pearson_val, pearson_p)
print(round(pearson_val, 2))
spearman_val, spearman_p = spearmanr(filtered['implicit_cosine'], filtered[y])
print(round(spearman_val, 2), spearman_p)

print(filtered[['implicit_cosine', 'num_ratings', y, 'name']].sort_values('implicit_cosine'))


#%%
names = {
    'name': 'Name', 'num_ratings': '# Ratings', 
    'like-boycott-ratio': 'Similar User Effect Ratio',
    'like-boycott-val':  '% change surfaced hits, similar users',
    'not-like-val':  '% change surfaced hits, other users',
}

table2_df = df[['name', 'num_ratings', 'like-boycott-val', 'not-like-val', 'like-boycott-ratio']].rename(index=str, columns=names)

table2_df.to_csv('table2.csv', index=False, float_format='%.2f', columns=[
        'Name', '# Ratings', 
        names['like-boycott-val'],
        names['not-like-val'],
        'Similar User Effect Ratio'
])

print(table2_df)


