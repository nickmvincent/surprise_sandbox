import pandas as pd
import json

# fn = 'standard_results/ml-1m_ratingcv_standards_for_SVD_10.csv'
for fn in [
    #'standard_results/ml-20m_ratingcv_standards_for_MovieMean.csv',
    #'standard_results/ml-20m_ratingcv_standards_for_GlobalMean.csv',
    #'standard_results/ml-20m_ratingcv_standards_for_GuessThree.csv',
    #'standard_results/ml-20m_ratingcv_standards_for_SVD.csv',
    'standard_results/ml-20m_ratingcv_standards_for_KNNBaseline_item_msd.csv',
]:
    
    df = pd.read_csv(fn, header=None)

    print(df)

    d = {}
    for i, row in df.iterrows():
        print(row)
        col = row[0]
        val = row[1]
        print(col, val)
        d[col] = val

    with open(fn.replace('.csv', '.json'), 'w') as f:
        json.dump(d, f)