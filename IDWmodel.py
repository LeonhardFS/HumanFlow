# Merkozy - best model so far (achieves 8.40477 on Leaderboard)

import numpy as np
from scipy.stats.mstats import mode
from copy import deepcopy
import pandas as pd
import time
from helper import *

# compute weights based on distance
def compute_invdist_weights(adjacency_list):
	if model_mode == 'full':
		df_sensors = pd.read_csv('data/sensor-coordinates.txt')
	else:
		df_sensors = pd.read_csv('data-final/sensor-coordinates.txt')
	df_sensors.columns = ['SID', 'X', 'Y']

	weight_list = {}
	for key in adjacency_list.keys():
	    weight_list[key] = []
	    b = np.array([df_sensors.loc[key - 1].X, df_sensors.loc[key - 1].Y])
	    for el in adjacency_list[key]:
	        # manhattan distance
	        a = np.array([df_sensors.loc[el - 1].X, df_sensors.loc[el - 1].Y])
	        dist = np.linalg.norm(a - b, ord=1)

	        # inverse distance weighting
	        weight_list[key].append(1.0 / dist)

	    # scale it accordingly
	    weight_list[key] = np.array(weight_list[key])
	    weight_list[key] = weight_list[key] / np.sum(weight_list[key])
	return weight_list

# Several possibilities to average the result among the neighbors:
#    - cumulative sum of neighbors (v1)
#    - average (truncated) (v2)
#    - average (raw) (v3) 
#    - average (rounded) (v4)
#    - average with inv distance weight, larger adj list (raw) (v5 Leo) ** best (27. euclidean)
def fill_neighbors(row, col_name, adjacency_list, weight_list):
    if row[col_name] == -1:
        new_value = 0.
        count = 0
        col_ind = int(col_name[1:])
        pos = 0
        for n in adjacency_list[col_ind]:
            if row[n] != -1:
                new_value += row[n] * weight_list[col_ind][pos]
                count += weight_list[col_ind][pos]
            pos = pos + 1
            
        # if some neighbours were working return their average (raw)
        if count:
            return round(new_value / (1. * count))
        
        # if no neighbours were working, return 0 (most frequent value)
        return new_value
    else:
        return row[col_name]


def IDWmodel(df_train):
	print 'computing adjacency list...'
	adjacency_list = compute_adjlist(27.) # best model

	print 'computing weights based on distance...'
	weight_list = compute_invdist_weights(adjacency_list)

	# make prediction
	print 'starting prediction...'
	col_names = ['S'+str(i) for i in xrange(1, 57)]
	df_train_neighbors_avg = df_train.copy()
	total_time = 0.
	for col_name in col_names:
	    start = time.time()
	    df_train_neighbors_avg[col_name] = df_train.apply(lambda row: fill_neighbors(row, col_name, adjacency_list, weight_list),axis=1)
	    col_time = time.time() - start
	    total_time += col_time
	    print 'Col {} computed in {:.2f}s, total: {:.2f}s'.format(col_name, col_time, total_time)

	print '--> finished in {:.2f}s'.format(total_time)
	return df_train_neighbors_avg


def build_IDWmodel():
	df_train = load_train_data()

	# fit model
	df_train_neighbors_avg = IDWmodel(df_train)

	print 'performing check...'
	# Checking that all the values are filled (i.e no -1 left)
	cum_sum = 0
	col_names = ['S'+str(i) for i in xrange(1, 57)]
	for col in col_names:
	    cum_sum += len(df_train_neighbors_avg[df_train_neighbors_avg[col] == -1])
	assert(cum_sum < 0.001)
	print 'writing to file...'
	if model_mode == 'full':
		create_submission_file(df_train_neighbors_avg, 'models-final/IDWmodel.csv')
	else:
		create_submission_file(df_train_neighbors_avg, 'models/IDWmodel.csv')

	# save model also in data folder
	if model_mode == 'full':
		df_train_neighbors_avg.to_csv('data-final/IDWmodel-final.csv', index=False)
	else:
		df_train_neighbors_avg.to_csv('data/IDWmodel_train.csv', index=False)
	print 'done!'

# main code
if __name__ == "__main__":
	build_IDWmodel()
