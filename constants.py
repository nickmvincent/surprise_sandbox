"""
Put constants here that are used throughout the project.

Do not include other dependencies or imports here.
Those should go in specs.py
"""

MEASURES = [
    'RMSE', 'MAE', 'list_metrics'
]

threshold = 4
LIST_METRICS = [
    'prec10t{}'.format(threshold),
    'prec5t{}'.format(threshold),
    'rec10t{}'.format(threshold),
    'rec5t{}'.format(threshold),
    'ndcg10',
    'ndcg5',
    'ndcgfull',
    'hits',
    'normhits',
    'avgrating',
    'avgest',
    'falsepos',
    'totalhits',
]

def get_metric_names():
    metric_names = []
    for measure in MEASURES[:2]:
        metric_names.append(measure.lower())
    for name in LIST_METRICS:
        #name = measure.lower()
        #metric_names.append(name)
        #metric_names.append('tail' + name)
        metric_names += [
            name, 'tail' + name, name + '_frac', 'tail' + name + '_frac',
        ]

    return metric_names
