## Linear model with moving average
# tune on moving average window
import numpy as np
from scipy.stats.mstats import mode
from copy import deepcopy
import pandas as pd
import time
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from helper import *
from IDWmodel import *

import sklearn


# ### Building a reference table with average daily value of the sensor
# specify with num_minutes over how many minutes the time should be bucketed
def build_avg_time_table(df_train, num_minutes=5):
    
    # different averaging does not seem to have an effect...
    df_train['day_time'] = (df_train['time'] % (100 * 100)) // 100 * 100 + ((df_train['time'] % 100) // num_minutes) * num_minutes
    
    # Initializing the dataframe
    # Update: rounding the value
    col_name = 'S1'
    df_day_avg_values = df_train[[col_name, 'day_time']][df_train[col_name] != -1].groupby('day_time').mean().apply(pd.Series.round)

    col_names = ['S'+str(i) for i in xrange(1, 57)]
    for col_name in col_names[1:]:
        df_day_avg_values = df_day_avg_values.join(df_train[[col_name, 'day_time']][df_train[col_name] != -1].groupby('day_time').mean().apply(pd.Series.round))
        
    return df_day_avg_values

# Args:
#    window_sizes = list of window_size for the moving sum 

def prediction_augmented(df_train, col_names, df_day_avg_values, adjacency_list, df_model, prediction_model, window_sizes=[10], do_rounding = False):
    staircaseA_nodes = ['S42', 'S46']
    staircaseB_nodes = ['S34', 'S35']
    staircaseC_nodes = ['S52', 'S53']
    
    # Dataframe to store the model prediction
    df_model_lr = df_model.copy()

    # Dataframe storing the moving neighbors values
    window_features = df_model[['time']].copy()
    
    for window_size in window_sizes:
    # Building the moving sum for the features before/after for each neighbor
        model_curr_before = pd.rolling_sum(df_model.sort_index(ascending=False), window_size+1) - df_model
        model_curr_after = pd.rolling_sum(df_model, window_size+1) - df_model
        model_curr_before = model_curr_before.rename(columns={col_:col_+'before'+str(window_size) for col_ in col_names})
        model_curr_after = model_curr_after.rename(columns={col_:col_+'after'+str(window_size) for col_ in col_names})
        temp = model_curr_after.join(model_curr_before[[col_+'before'+str(window_size) for col_ in col_names]])
        window_features = window_features.join(temp, lsuffix='left')
    
    for col in col_names:
        # X will store the features and the outcome Y
        X = df_train.copy()
        X = X.rename(columns={col:'Y'})
        X = pd.merge(X, df_day_avg_values[[col]], left_on='day_time', right_index=True)
        X = X.rename(columns={col:col+'avg'})

        # Building the neighbors (from adjacency list) with missing values filled as in model
        neighbors_col = ['S'+str(n) for n in adjacency_list[int(col[1:])]]
        
        # Name of the retained columns from window_features
        window_features_name = []
        for window_size in window_sizes:
            window_features_name += [col_+'before'+str(window_size) for col_ in neighbors_col] + [col_+'after'+str(window_size) for col_ in neighbors_col]
        
        X = X[['Y']].join(df_model[neighbors_col])
        X = X.join(window_features[window_features_name])
        # Removing the first and last element impossible to compute given the window_size
        X = X.sort_index()[max(window_sizes): - max(window_sizes)]
        
        # augment with staircase info
        X['sA'] = (col in staircaseA_nodes) * 1.
        X['sB'] = (col in staircaseB_nodes) * 1.
        X['sC'] = (col in staircaseC_nodes) * 1.

        X_train = X[X['Y'] != -1]
        X_test = X[X['Y'] == -1]
        test_indices = X[X['Y'] == -1].index
        col_values = df_model_lr[col]

        if len(X_test):
            # Models
            prediction_model = prediction_model.fit(X_train.drop('Y', axis=1), X_train.Y)
            col_values.ix[test_indices] = prediction_model.predict(X_test.drop('Y', axis=1))

            # Filling the result with the current sensor prediction
            if do_rounding:
                df_model_lr[col] = np.round(col_values)
            else:
                df_model_lr[col] = col_values
    return df_model_lr


# main step, build the model
if __name__ == '__main__':
    print 'loading train data...'
    df_train = load_train_data()
    if model_mode == 'full':
        idwmodel_file = 'data-final/IDWmodel-final.csv'
        submit_file = 'models-final/lr_model_ma_model_v2-final.csv'
    else:
        idwmodel_file = 'data/IDWmodel_train.csv'
        submit_file = 'models/lr_model_ma_model_v2.csv'

    # check if IDW model already exists, if not train it!
    if not file_exists(idwmodel_file):
        print 'building IDW model first...'
        df_IDWmodel = build_IDWmodel()
    else:
        print 'loading IDW model...'
        df_IDWmodel = pd.read_csv(idwmodel_file)


    ## tune here maybe...

    print 'building time table...'
    df_day_avg_values = build_avg_time_table(df_train)


    print 'computing adj list...'
    col_names = ['S'+str(i) for i in xrange(1, 57)]
    adjacency_list = compute_adjlist(27)

    ## tune here
    ## Specify here the classification model
    clf = linear_model.LassoLarsCV(positive=True, max_iter=1500)

    print 'computing linear model...'

    ## tune here
    num_rounds = 8
    assert (num_rounds >= 2)
    start = time.time()
    total_time = 0.
    df_model_lr = prediction_augmented(df_train, col_names, df_day_avg_values, adjacency_list, df_IDWmodel, clf, window_sizes=[10])
    cur_time = time.time() - start
    total_time += cur_time
    print 'round #1 on IDW model, {:.2f}/{:.2f}s ...'.format(cur_time, total_time * num_rounds)

    ## tune here
    # First prediction with small windows (1/4 of the number of rounds)
    # To have good prediction on the first and last rows
    for i in xrange((num_rounds - 2)/4):
    	start = time.time()
    	df_model_lr = prediction_augmented(df_train, col_names, df_day_avg_values, adjacency_list, df_model_lr, clf, window_sizes=[10])
    	cur_time = time.time() - start
    	total_time += cur_time
    	print 'round #{}, {:.2f}/{:.2f}s ...'.format(i+2, cur_time, total_time  * num_rounds / (i + 2) )

    ## tune here
    # Second prediction with small windows (1/4 of the number of rounds)
    # To have good prediction on the first and last rows
    mid_window_sizes = [10, 15]
    for i in xrange((num_rounds - 2)/4):
        start = time.time()
        df_model_lr = prediction_augmented(df_train, col_names, df_day_avg_values, adjacency_list, df_model_lr, clf, window_sizes=mid_window_sizes)
        cur_time = time.time() - start
        total_time += cur_time
        print 'round #{}, {:.2f}/{:.2f}s ...'.format(i+2, cur_time, total_time  * num_rounds / (i + 2) )

    ## tune here
    # Third prediction with small windows (1/2 of the number of rounds)
    # To have good prediction on the first and last rows
    large_window_sizes = [10, 15, 20, 30]
    for i in xrange((num_rounds - 2)/2):
        start = time.time()
        df_model_lr = prediction_augmented(df_train, col_names, df_day_avg_values, adjacency_list, df_model_lr, clf, window_sizes=large_window_sizes)
        cur_time = time.time() - start
        total_time += cur_time
        print 'round #{}, {:.2f}/{:.2f}s ...'.format(i+2, cur_time, total_time  * num_rounds / (i + 2) )

    start = time.time()
    df_model_lr = prediction_augmented(df_train, col_names, df_day_avg_values, adjacency_list, df_model_lr, clf, window_sizes=large_window_sizes, do_rounding = True)
    cur_time = time.time() - start
    total_time += cur_time
    print 'round #{} with rounding, {:.2f}s ...'.format(3 + num_rounds, cur_time)
    print '--> finished in {:.2f}s'.format(total_time)

    print 'writing to file...'
    create_submission_file(df_model_lr, submit_file)
    print 'done!'

