from surprise.prediction_algorithms.predictions import Prediction
import ast

with open(
    '{}_seed0_fold{}_predictions.txt'.format('predictions/standards/test_ml-1m_GlobalMean_', 0), 'r'
) as file_handler:
    content = ['[' + x.strip('\n') + ']' for x in file_handler.readlines()]
    assert(content[0] == '[uid,iid,r_ui,est,details,crossfold_index]')
    predictions = [Prediction(*ast.literal_eval(line)[:-1]) for line in content[1:]]
    print(predictions)
