"""
Control specs here that affect many scripts throughout the project
"""
from surprise import SVD, Dataset, KNNBaseline, GuessThree, GlobalMean, MovieMean, KNNBasic, TwentyMean

# Notes on default algo params:
# KNN uses 40 max neighbors by default
# min neighbors is 1, and if there's no neighbors then use global average!
# default similarity is MSD
# default user or item is USER-BASED but we override that.

ALGOS = {
    'SVD': SVD(random_state=0),
    # 'SVD50': SVD(n_factors=50),
    #'KNNBasic_user_msd': KNNBasic(sim_options={'user_based': True}),
    # 'KNNBasic_user_cosine': KNNBasic(sim_options={'user_based': True, 'name': 'cosine'}),
    # 'KNNBasic_user_pearson': KNNBasic(sim_options={'user_based': True, 'name': 'pearson'}),
    # 'KNNBasic_item_msd': KNNBasic(sim_options={'user_based': False}),
    # 'KNNBasic_item_cosine': KNNBasic(sim_options={'user_based': False, 'name': 'cosine'}),
    # 'KNNBasic_item_pearson': KNNBasic(sim_options={'user_based': False, 'name': 'pearson'}),
    #'KNNBaseline_item_msd': KNNBaseline(sim_options={'user_based': False}),
}

ALGOS_FOR_STANDARDS = {
    # flag: uncomment when done
    #'KNNBasic_user_msd': KNNBasic(sim_options={'user_based': True}),
    #'KNNBasic_user_cosine': KNNBasic(sim_options={'user_based': True, 'name': 'cosine'}),
    #'KNNBasic_user_msd_10': KNNBasic(sim_options={'user_based': True}, k=10),
    #'KNNBasic_user_cosine_10': KNNBasic(sim_options={'user_based': True, 'name': 'cosine'}, k=10),
    #'KNNBasic_item_msd': KNNBasic(sim_options={'user_based': False}),
    #'KNNBasic_item_cosine': KNNBasic(sim_options={'user_based': False, 'name': 'cosine'}),
    'SVD': SVD(random_state=0),
    #'KNNBaseline_item_msd': KNNBaseline(sim_options={'user_based': False}),
    #'GuessThree': GuessThree(),
    #'GlobalMean': GlobalMean(),
    #'MovieMean': MovieMean(),
    #'TwentyMean': TwentyMean(),
}


NUM_FOLDS = 5
