from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.model_selection import train_test_split,StratifiedKFold
import gcforest.data_load as load
import pandas as pd
import json
import os
import os.path as osp


def save_features(data_name, features):
    output_dir = osp.join("output", "reliefF")
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    file = osp.join(output_dir, data_name)
    with open(file, 'w') as wf:
        wf.write(json.dumps(features))


cv = StratifiedKFold(n_splits=5, random_state=0)

data_sets = ['cirrhosis','obesity']

for data_name in data_sets:
    X, y = None,None

    if data_name == 'obesity':
        X, y = load.obesity_data()
    elif data_name == 'cirrhosis':
        X, y = load.cirrhosis_data()
    else:
        X, y = load.t2d_data()
    i = 0
    for tr,te in cv.split(X,y):
        X_train, X_test, y_train, y_test = X.iloc[tr], X.iloc[te], y[tr], y[te]
        fs = ReliefF(n_features_to_select=300)
        fs.fit(X_train.values, y_train)


        columns_names = X.columns.tolist()
        features = []
        scores = []


        for feature_name, feature_score in zip(columns_names, fs.feature_importances_):
            features.append(columns_names.index(feature_name))
            scores.append(feature_score)

        se = pd.Series(data=scores,index=features)
        save_features(data_name + "-cv-" + str(i), se.to_dict())
        i = i +1