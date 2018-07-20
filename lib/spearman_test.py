import json
import os
import os.path as osp
import pandas as pd
from scipy.stats import spearmanr
import numpy as  np
import gcforest.data_load as load
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.model_selection import train_test_split

def consistency_index(sel1, sel2, num_features):
    """ Compute the consistency index between two sets of features.
    Parameters
    ----------
    sel1: set
        First set of indices of selected features
    sel2: set
        Second set of indices of selected features
    num_features: int
        Total number of features
    Returns
    -------
    cidx: float
        Consistency index between the two sets.
    Reference
    ---------
    Kuncheva, L.I. (2007). A Stability Index for Feature Selection.
    AIAC, pp. 390--395.
    """
    observed = float(len(sel1.intersection(sel2)))
    expected = len(sel1) * len(sel2) / float(num_features)
    maxposbl = float(min(len(sel1), len(sel2)))
    cidx = -1.
    # It's 0 and not 1 as expected if num_features == len(sel1) == len(sel2) => observed = n
    # Because "take everything" and "take nothing" are trivial solutions we don't want to select
    if expected != maxposbl:
        cidx = (observed - expected) / (maxposbl - expected)
    return cidx


def consistency_index_k(sel_list, num_features):
    """ Compute the consistency index between more than 2 sets of features.
    This is done by averaging over all pairwise consistency indices.
    Parameters
    ----------
    sel_list: list of lists
        List of k lists of indices of selected features
    num_features: int
        Total number of features
    Returns
    -------
    cidx: float
        Consistency index between the k sets.
    Reference
    ---------
    Kuncheva, L.I. (2007). A Stability Index for Feature Selection.
    AIAC, pp. 390--395.
    """
    cidx = 0.
    for k1, sel1 in enumerate(sel_list[:-1]):
        # sel_list[:-1] to not take into account the last list.
        # avoid a problem with sel_list[k1+1:] when k1 is the last element,
        # that give an empty list overwise
        # the work is done at the second to last element anyway
        for sel2 in sel_list[k1+1:]:
            cidx += consistency_index(set(sel1), set(sel2), num_features)
    cidx = 2.  * cidx / (len(sel_list) * (len(sel_list) - 1))
    return cidx


data_sets = ["cirrhosis", "t2d", "obesity"]
feature_sets = [50, 100, 150, 200, 300]
feature_len = [542, 572, 465]


for k, data_name in enumerate(data_sets):
    print(data_name)
    for feat in feature_sets:
        print("===================================")
        output_dir = osp.join("output", "result")
        ma = []
        for i in range(5):
            path = osp.join(output_dir, data_name +"-cv-" + str(i))
            file = open(path, 'r')
            dicts = json.load(file)

            df = pd.Series(dicts)
            df = df.sort_values(ascending=False)
            ma.append(df.index.tolist()[0:feat])

        sp = consistency_index_k(ma, feature_len[k])
        print(sp)

        ######reliefF
        output_dir = osp.join("output", "reliefF")
        ma = []
        for i in range(5):
            path = osp.join(output_dir, data_name +"-cv-" + str(i))
            file = open(path, 'r')
            dicts = json.load(file)

            df = pd.Series(dicts)
            df = df.sort_values(ascending=False)
            ma.append(df.index.tolist()[0:feat])

        sp = consistency_index_k(ma, feature_len[k])
        print(sp)

        #####SVM-REF
        output_dir = osp.join("output", "svm-ref")
        ma = []
        for i in range(5):
            path = osp.join(output_dir, data_name+"-cv-" + str(i))
            file = open(path, 'r')
            dicts = json.load(file)

            df = pd.Series(dicts)
            df = df[df == 1]
            ma.append(df.index.tolist()[0:feat])

        sp = consistency_index_k(ma, feature_len[k])
        print(sp)
