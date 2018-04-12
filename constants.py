"""
Put constants here that are used throughout the project.

Do not include other dependencies or imports here.
Those should go in specs.py
"""


ALGO_NAMES = [
    'SVD',
    'KNNBaseline_item_msd',
]

MEASURES = [
    'RMSE', 'MAE', 'prec10t4_prec5t4_rec10t4_rec5t4_ndcg10_ndcg5_ndcgfull'
]

def get_metric_names():
    metric_names = []
    for measure in MEASURES:
        if '_' in measure:
            metric_names += measure.lower().split('_')
        else:
            metric_names.append(measure.lower())
    return metric_names
