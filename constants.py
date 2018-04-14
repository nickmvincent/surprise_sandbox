"""
Put constants here that are used throughout the project.

Do not include other dependencies or imports here.
Those should go in specs.py
"""

ALGO_NAMES = [
    'SVD',
    #'KNNBasic_user_msd'    
    'KNNBaseline_item_msd',
]

MEASURES = [
    'RMSE', 'MAE', 'prec10t4_prec5t4_rec10t4_rec5t4_ndcg10_ndcg5_ndcgfull'
]

def get_metric_names():
    metric_names = []
    for measure in MEASURES:
        if '_' in measure:
            splitnames = measure.lower().split('_')
            metric_names += splitnames
            metric_names += ['tail' + x for x in splitnames]
        else:
            metric_names.append(measure.lower())
    return metric_names
