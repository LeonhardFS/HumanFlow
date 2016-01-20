import numpy as np
import pandas as pd

# converts time stamp as int in DHHMM into index for the table
def timestamptoindex(ts):
    mm = ts % 100
    hh = (ts % (100 * 100)) // 100
    d = ts // (100 * 100)
    
    index = mm + 60 * hh + 60 * 24 * (d - 1)
    return index

# loads train data
def load_train_data():
    df_train = pd.read_csv('data/train.txt', skipinitialspace=True)
    df_train.rename(columns={'Timestamp (DHHMM)':'time'}, inplace=True)
    return df_train

def get_count(row, train):
    start = int(row['Start Time'])
    end = int(row['End Time'])
    col = row['Sensor ID']
    i0 = timestamptoindex(start)
    i1 = timestamptoindex(end)

    cum_sum = train[col][i0:i1].sum()
    return cum_sum

# #### To build a submission file
def create_submission_file(df, filename):

    df_test = pd.read_csv('data/test.txt', skipinitialspace=True)
    df_test['Count'] = df_test.apply(lambda row: get_count(row, df),axis=1)

    # Saving it under the name submission_name
    submission_name = 'dummy.csv'
    df_test['Index'] = df_test.index + 1
    df_test[['Index', 'Count']].to_csv(filename, index=False)

# Compute the adjacency list with threshold from the dataframe df
def compute_adjlist(threshold, df):
    # computing the adjacency list based on distance
    for key in xrange(56):
        node = key + 1
        adjacency_list[node] = []
        # go through all other nodes, if distance is below threshold, fine!
        for other_key in xrange(56):
            if other_key == key:
                continue

            a = np.array([df.loc[key].X, df.loc[key].Y])
            b = np.array([df.loc[other_key].X, df.loc[other_key].Y])
            dist = np.linalg.norm(a - b, ord=1)

            if dist < threshold:
                adjacency_list[node].append(other_key + 1)
                
    return adjacency_list
