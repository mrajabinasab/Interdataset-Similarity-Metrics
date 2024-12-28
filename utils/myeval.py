import numpy as np
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from clustpy.metrics.clustering_metrics import unsupervised_clustering_accuracy
from sklearn.cluster import KMeans
from statistics import mean
from utils.myfeatureselector import selector_tuned


warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model._logistic")


def unsupervised_eval(data, labels, method, target_features_num, avg_steps=10):
    selfeats = selector_tuned(data, labels, method, target_features_num)
    new_data = data.iloc[:, selfeats]
    k = len(np.unique(labels))
    clsacc = np.empty(avg_steps)
    for i in range(avg_steps):
        kmeans = KMeans(n_clusters=k, random_state=i, n_init="auto").fit(new_data)
        clsacc[i] = unsupervised_clustering_accuracy(labels, kmeans.labels_)
    return clsacc.mean()


def supervised_eval(X, y, method, target_features_num, avg_steps = 10, classifiers=["dt"], test_ratio=0.2):
    if method == "no_change":
        new_X = X
    else:
        selfeats = selector_tuned(X, y, method, target_features_num)
        new_X = X.iloc[:, selfeats]
    result = []
    if len(np.unique(y)) > 2:
        myscoring = ('f1_macro')
    else:
        myscoring = ('f1')
    for i in range(avg_steps):
        for mclf in classifiers:
            if mclf == 'dt':
                clf = DecisionTreeClassifier(random_state=i)
            if mclf == 'lr':
                clf = LogisticRegression(random_state=i, max_iter=1000, solver='saga')
            if mclf == 'knn':
                clf = KNeighborsClassifier(n_neighbors=5)
            if mclf == 'svm':
                clf = SVC(kernel='rbf', random_state=i)
            scores = cross_validate(clf, new_X, y, scoring=myscoring, cv=5)
            result.append(scores['test_score'].mean())
    return mean(result)