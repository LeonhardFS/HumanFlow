## linear recursive model, jumps to 5.56

import numpy as np
from scipy.stats.mstats import mode
from copy import deepcopy
import pandas as pd
import time
from helper import *
from IDWmodel import *
from sklearn.linear_model import LinearRegression

# ### Building a reference table with average daily value of the sensor
def build_avg_time_table(df_train):
    df_train['day_time'] = df_train.time % 10000

    # Initializing the dataframe
    # Update: rounding the value
    col_name = 'S1'
    df_day_avg_values = df_train[[col_name, 'day_time']][df_train[col_name] != -1].groupby('day_time').mean().apply(pd.Series.round)

    col_names = ['S'+str(i) for i in xrange(1, 57)]
    for col_name in col_names[1:]:
        df_day_avg_values = df_day_avg_values.join(df_train[[col_name, 'day_time']][df_train[col_name] != -1].groupby('day_time').mean().apply(pd.Series.round))
        
    return df_day_avg_values

def lr_prediction(df_train, col_names, df_day_avg_values, adjacency_list, df_model):
    # Dataframe to store the model prediction
    df_model_lr = df_train.copy()
    for col in col_names:
        # X will store the features and the outcome Y
        X = df_train.copy()
        X = X.rename(columns={col:'Y'})
        X = pd.merge(X, df_day_avg_values[[col]], left_on='day_time', right_index=True)
        X = X.rename(columns={col:col+'avg'})

        # Building the neighbors (from adjacency list) with missing values filled as in model
        neighbors_col = ['S'+str(n) for n in adjacency_list[int(col[1:])]]
        X = X[['Y']].join(df_model[neighbors_col])

        X_train = X[X['Y'] != -1]
        X_test = X[X['Y'] == -1]
        test_indices = X[X['Y'] == -1].index
        col_values = X['Y']

        if len(X_test):
            # Models
            lr = LinearRegression()
            lr = lr.fit(X_train.drop('Y', axis=1), X_train.Y)
            col_values.ix[test_indices] = lr.predict(X_test.drop('Y', axis=1))

            # Filling the result with the current sensor prediction
            df_model_lr[col] = np.round(col_values)
    return df_model_lr


### train the model, main code here
if __name__ == "__main__":
    df_train = load_train_data()
    if model_mode == 'full':
        idwmodel_file = 'data-final/IDWmodel-final.csv'
        submit_file = 'models-final/lr_model-final.csv'
    else:
        idwmodel_file = 'data/IDWmodel_train.csv'
        submit_file = 'models/lr_model.csv'

    # check if IDW model already exists, if not train it!
    if not file_exists(idwmodel_file):
    	print 'building IDW model first...'
    	df_IDWmodel = build_IDWmodel()
    else:
    	print 'loading IDW model...'
    	df_IDWmodel = pd.read_csv(idwmodel_file)


    print 'computing time features...'
    df_day_avg_values = build_avg_time_table(df_train)

    print 'computing adj list...'
    adjacency_list = compute_adjlist(27.)

    col_names = ['S'+str(i) for i in xrange(1, 57)]
    print 'running linear model, round #1 ...'
    df_model_lr = lr_prediction(df_train, col_names, df_day_avg_values, adjacency_list, df_IDWmodel) # 5.78
    print 'running linear model, round #2 ...'
    df_model_lr = lr_prediction(df_train, col_names, df_day_avg_values, adjacency_list, df_model_lr) # 5.56

    print 'writing to file...'
    create_submission_file(df_model_lr, submit_file)
    print 'done!'




