import pandas as pd
import json

# fn = 'standard_results/ml-1m_ratingcv_standards_for_SVD_10.csv'
fn = 'standard_results/ml-1m_ratingcv_standards_for_KNNBasic_item_msd.json'
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