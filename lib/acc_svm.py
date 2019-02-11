import json
import numpy as np
import os.path as osp
import pickle
import gcforest.data_load as load
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

data_sets = ["cirrhosis", 'obesity', 't2d']
feature_sets = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# feature_len = [542, 572, 465]

clf_svm = SVC(kernel='linear')
cv = StratifiedKFold(n_splits=5, shuffle=False, random_state=0)

for k, data_name in enumerate(data_sets):
    print("==================")
    print(data_name)
    ll = []

    X, y = None, None

    if data_name == 'obesity':
        X, y = load.obesity_data()
    elif data_name == 'cirrhosis':
        X, y = load.cirrhosis_data()
    else:
        X, y = load.t2d_data()

    clf_acc_before = cross_val_score(clf_svm, X, y, cv=cv, scoring='accuracy')
    print(np.mean(clf_acc_before))

    for feat in feature_sets:
        llm = []
        print("------------")
        # ###### deep forest
        output_dir = osp.join("output", "result")
        mat = []
        for i in range(5):
            path = osp.join(output_dir, data_name + "-cv-" + str(i))
            file = open(path, 'r')
            dicts = json.load(file)

            df = pd.Series(dicts)
            df = df.sort_values(ascending=False)

            li = list(map(int, df.index.values.tolist()))
            X_hat = X.iloc[:, li[0:feat]]
            clf_acc_after = cross_val_score(clf_svm, X_hat, y, cv=cv, scoring='accuracy')
            clf_acc_df = np.mean(clf_acc_after)
            mat.append(clf_acc_df)
        llm.append(np.mean(mat))
        ######reliefF
        output_dir = osp.join("output", "reliefF")
        ma = []
        for i in range(5):
            path = osp.join(output_dir, data_name + "-cv-" + str(i))
            with open(path, 'rb') as file:
                li = pickle.load(file)
            X_hat = X.iloc[:, li[0:feat]]
            clf_acc_after = cross_val_score(clf_svm, X_hat, y, cv=cv, scoring='accuracy')
            ma.append(clf_acc_after)
        llm.append(np.mean(ma))

        ######mRMR
        output_dir = osp.join("output", "mRMR")
        ma = []
        for i in range(5):
            path = osp.join(output_dir, data_name + "-cv-" + str(i))
            with open(path, 'rb') as file:
                li = pickle.load(file)
            X_hat = X.iloc[:, li[0:feat]]
            clf_acc_after = cross_val_score(clf_svm, X_hat, y, cv=cv, scoring='accuracy')
            ma.append(clf_acc_after)
        llm.append(np.mean(ma))

        ####SVM-RFE
        output_dir = osp.join("output", "svm-ref")
        ma = []
        for i in range(5):
            path = osp.join(output_dir, data_name + "-cv-" + str(i))
            with open(path, 'r') as file:
                dicts = json.load(file)

            df = pd.Series(dicts)
            df = df.sort_values(ascending=True)
            li = df.index.tolist()[0:feat]
            li = list(map(int, df.index.tolist()[0:feat]))
            X_hat = X.iloc[:, li]

            clf_acc_after = cross_val_score(clf_svm, X_hat, y, cv=cv, scoring='accuracy')
            ma.append(clf_acc_after)
        llm.append(np.mean(ma))

        ll.append(llm)
    dd = pd.DataFrame(ll, columns=['Deep Forest', 'ReliefF', 'mRMR', 'SVM-RFE'],
                      index=['5', '10', '15', '20', '25', '30', '35', '40', '45', '50'])
    dd.to_csv(data_name + "-SVM-ACC")
