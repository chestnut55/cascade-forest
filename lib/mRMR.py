import pandas as pd
import pymrmr
from sklearn.model_selection import StratifiedKFold
import gcforest.data_load as load
import os
import os.path as osp
import pickle

def save_features(data_name, features):
    output_dir = osp.join("output", "mRMR")
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    file = osp.join(output_dir, data_name)
    with open(file, 'wb') as wf:
        pickle.dump(features, wf)

# df = pd.read_csv('test_colon_s3.csv')
# pymrmr.mRMR(df, 'MIQ', 10)
cv = StratifiedKFold(n_splits=5, random_state=0)
data_sets = ['cirrhosis','obesity','t2d']


for data_name in data_sets:
    X, y = None,None

    kun_index_list = []
    if data_name == 'obesity':
        X, y = load.obesity_data()
    elif data_name == 'cirrhosis':
        X, y = load.cirrhosis_data()
    else:
        X, y = load.t2d_data()
    i = 0
    for tr, te in cv.split(X, y):
        X_train, X_test, y_train, y_test = X.iloc[tr], X.iloc[te], y[tr], y[te]
        X_train.insert(loc = 0, column='class', value = y_train)

        features_list = list(X.columns.values)
        feas = pymrmr.mRMR(X_train, 'MIQ', 5)
        index = []
        for fea in feas:
            idx = features_list.index(fea)
            index.append(idx)

        save_features(data_name + "-cv-" + str(i), idx)
        i = i + 1