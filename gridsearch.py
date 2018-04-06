import random

from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import GridSearchCV


def main():
    # Load the full dataset.
    data = Dataset.load_builtin('ml-1m')

    # Select your best algo with grid search.
    print('Grid Search...')
    param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005]}
    grid_search = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
    grid_search.fit(data)

    algo = grid_search.best_estimator['rmse']
    print(algo)

if __name__ == '__main__':
    main()