import random
import pandas as pd
from feature.selector import Selective, SelectionMethod
from sklearn.preprocessing import MinMaxScaler


def selector_tuned(X, y, method, target_features_num):
    if target_features_num == len(X.columns):
        return list(range(len(X.columns)))
    if method == "chi_square":
        selector = Selective(SelectionMethod.Statistical(target_features_num, method="chi_square"))
    if method == "mutual_info":
        selector = Selective(SelectionMethod.Statistical(target_features_num, method="mutual_info"))
    if method == "anova":
        selector = Selective(SelectionMethod.Statistical(target_features_num, method="anova"))
    if method == "linear":
        selector = Selective(SelectionMethod.Linear(target_features_num, regularization="none"))
    if method == "lasso":
        selector = Selective(SelectionMethod.Linear(target_features_num, regularization="lasso", alpha=1000))
    if method == "ridge":
        selector = Selective(SelectionMethod.Linear(target_features_num, regularization="ridge", alpha=1000))
    if method == "random_forest":
        selector = Selective(SelectionMethod.TreeBased(target_features_num))
    if method == "random":
        selfeats = []
        for i in range(target_features_num):
            sel = random.randint(0, len(X.columns) - 1)
            while(sel in selfeats):
                sel = random.randint(0, len(X.columns)-1)
            selfeats.append(sel)
        return selfeats
    scl = MinMaxScaler()
    scl.fit(X)
    my_X = scl.transform(X)
    my_X = pd.DataFrame(my_X, index=X.index, columns=X.columns)
    subset = selector.fit_transform(my_X, y)
    selfeats = []
    for column in list(subset.columns):
        selfeats.append(X.columns.get_loc(column))
    return selfeats