import json
import os
import os.path as osp
import pandas as pd
from utils import avg_importance
import gcforest.data_load as load
import matplotlib.pyplot as plt
import numpy as np

data_sets = ["cirrhosis","obesity","t2d"]
for k, data_name in enumerate(data_sets):
    print(data_name)
    output_dir = osp.join("output", "result")
    ca_features = pd.Series()
    for i in range(5):
        path = osp.join(output_dir, data_name +"-cv-" + str(i))
        file = open(path, 'r')
        dicts = json.load(file)

        df = pd.Series(dicts)
        df = df.sort_values(ascending=False)
        ca_features = avg_importance(ca_features, df)
    ca_features = ca_features.sort_values(ascending=False)

    columns = ca_features.index.tolist()
    columns = list(map(int, columns))

    X, Y = None, None
    healthy_idx = []
    disease_idx = []
    if data_name == 'cirrhosis':
        X, Y = load.cirrhosis_data()
        for i in range(len(Y)):
            if Y[i] == 1:  # healthy
                healthy_idx.append(i)
            else:  # disease
                disease_idx.append(i)
    elif data_name == 'obesity':
        X, Y = load.obesity_data()
        for i in range(len(Y)):
            if Y[i] == 0:  # healthy
                healthy_idx.append(i)
            else:  # disease
                disease_idx.append(i)
    elif data_name == 't2d':
        X, Y = load.t2d_data()
        for i in range(len(Y)):
            if Y[i] == 0:  # healthy
                healthy_idx.append(i)
            else:  # disease
                disease_idx.append(i)
    X_hat = X.ix[:, columns]

    healthy = X_hat.ix[healthy_idx]
    disease = X_hat.ix[disease_idx]

    ca_features.to_csv(data_name+'_importance.csv')

    healty_mean = np.mean(healthy,axis=0)
    healty_mean.to_csv(data_name+'_healthy.csv')
    disease_mean = np.mean(disease, axis=0)
    disease_mean.to_csv(data_name + '_disease.csv')



