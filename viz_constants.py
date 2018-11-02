metric2title = {
    'ndcg10': 'NDCG@10',
    '2hits-prec5t4': 'Loss in Precision@5-estimated hits, pb=2',
    '4hits-prec5t4': 'Loss in Precision@5-estimated hits, pb=4',
    '2hits-ndcg10': 'Loss in ndcg@10-estimated hits, pb=2',
    'labor-hits-prec5t4': 'Loss in hits from data labor power',
    'consumer-hits-prec5t4': 'Loss in hits from consumer power',
    'rmse': 'RMSE',
    'prec10t4': 'Precision',
    'totalhits': 'Total Hits',
    'loghits': 'Log-Transformed Hits',
    'normhits': 'Fraction of Ideal Hits across Remaining Users',
    'biznormhits': 'Fraction of Ideal Hits across System'
}

group2scenario = {
    'all': 'Data Strike, users',
    'non-boycott': 'Data Boycott, users',
    'bizall': 'Data Strike, system',
    'biznon-boycott': 'Data Boycott, system'
}

num_users = {
    'ml-20m': 138493,
    'ml-1m': 6040
}
num_ratings = {
    'ml-20m': 20000263,
    'ml-1m': 1000209
}