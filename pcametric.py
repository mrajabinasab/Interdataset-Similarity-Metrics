import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def PCAMetric(realdata, fsdata, num_components = 1):
    r_scaled = StandardScaler().fit_transform(realdata)
    f_scaled = StandardScaler().fit_transform(fsdata)
    rpca = PCA(n_components=num_components)
    fpca = PCA(n_components=num_components)
    r_pca = rpca.fit_transform(r_scaled)
    f_pca = fpca.fit_transform(f_scaled)
    var_difference = sum(abs(rpca.explained_variance_ratio_- fpca.explained_variance_ratio_))
    len_r = np.sqrt(rpca.components_[0].dot(rpca.components_[0]))
    len_f = np.sqrt(fpca.components_[0].dot(fpca.components_[0]))
    rpca_comp_np = rpca.components_[0]
    angle_diff = min([np.arccos(rpca_comp_np @ (s*fpca.components_[0])) for s in [1,-1]])/(len_r*len_f)
    results = {'exp_var_diff': var_difference, 'comp_angle_diff': (angle_diff*2)/np.pi}
    return results, r_pca, f_pca


def AAD(X, selected_features):
    aad = 0
    not_selected_features = [q for q in range(len(X.columns)) if q not in selected_features]
    for p in not_selected_features:
        my_X = X.copy()
        my_X.iloc[:, p] = 0
        result, _, _ = PCAMetric(X, my_X)
        aad += result['comp_angle_diff']
    if len(not_selected_features) != 0:
        aad /= len(not_selected_features)
    else:
        aad = 0
    return aad