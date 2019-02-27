metric2title = {
    'ndcg10': 'NDCG@10',
    'tailndcg10': 'Tail NDCG@10',
    'ndcg5': 'NDCG@5',
    'ndcgfull': 'NDCG with All Items',
    '2hits-prec5t4': 'Loss in Precision@5-estimated hits, pb=2',
    '4hits-prec5t4': 'Loss in Precision@5-estimated hits, pb=4',
    '2hits-ndcg10': 'Loss in ndcg@10-estimated hits, pb=2',
    'labor-hits-prec5t4': 'Loss in hits from data labor power',
    'consumer-hits-prec5t4': 'Loss in hits from consumer power',
    'rmse': 'RMSE',
    'prec10t4': 'Precision@10',
    'tailprec10t4': 'Tail Precision@10',
    'prec5t4': 'Precision@5',
    'rec10t4': 'Recall@10',
    'tailrec10t4': 'Tail Recall@10',
    'rec5t4': 'Recall@5',
    'totalhits': 'Total Hits',
    'loghits': 'Log-Transformed Hits',
    'normhits': 'Norm Hits per User',
    'surfaced-hits': 'Surfaced Hits'
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
num_hits = {
    'ml-20m': 9995410,
    'ml-1m': 575281,
}