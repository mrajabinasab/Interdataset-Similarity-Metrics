import numpy as np

from typing import Tuple, List, Literal
from numpy import ndarray
from pandas import DataFrame

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import warnings

def _bound(value: float, range: Tuple[float, float]) -> float:
    """Bound a value between two values
    
    Args:
        range (Tuple[float, float]): The lower and upper bounds
        value (float): The value to bound
    
    Returns:
        float: The bounded value
    
    Example:
    >>> _bound(0.5,(0, 1))
    0.5
    """
    low, high = range
    return max(low, min(high, value))

def PCAMetric(data_base: ndarray | DataFrame, data_comp: ndarray | DataFrame, num_components: int = None, normalization: Literal['precise', 'approx'] = 'precise',  preprocess: Literal['std', 'mean'] = 'std') -> Tuple[dict, ndarray, ndarray]:
    """Function for claculating the difference in eigenvalues and eigenvectors 
    of the principal components of the two datasets.
    
    Args:
        data_base (array | DataFrame): The original data to use as a base
        data_comp (array | DataFrame): The data that are being compared to the base
        num_components (int): int, the number of components to consider for explained variance difference (default is all)
        normalization (Literal['precise', 'approx']): The normalization method used
            'precise': use the normalization factor as mentioned in the paper = d/(d+p-2)
            'approx': set the normalizaton factor to 0.5
        preprocess (Literal['std', 'mean']): The type of preprocessing to use
            'std': rescale by dividing standard deviation (default) 
            'mean': mean-subtracted data
    
    Returns:
        Dict[str, float]: The results of the comparison (explained variance difference and component angle difference)
        array: the projection of the base data
        array: the projection of the comparison data

    Example:
    >>> import pandas as pd
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> X = pd.DataFrame(iris.data, columns=iris.feature_names)
    >>> results, r_pca, f_pca = PCAMetric(X, X)
    >>> results['exp_var_diff']
    0.0
    >>> results['comp_angle_diff']
    0.0
    """
    if num_components is None:
        num_components = data_base.shape[1]
    if data_base.shape[1] == 1:
        raise ValueError("Cannot use a dataset with d = 1.")
    if num_components > data_base.shape[1]:
        warnings.warn("Number of principal components is set to a value larger than d. Automatically setting it to d.", UserWarning)
        num_components = data_base.shape[1]
    match normalization:
        case 'precise':
            factor = data_base.shape[1] / (data_base.shape[1] + num_components - 2)
        case 'approx':
            factor = 0.5
        case _:
            raise ValueError("Invalid normalization keyword")
            
    match preprocess:
        case 'std':
            b_scaled = StandardScaler().fit_transform(data_base)
            c_scaled = StandardScaler().fit_transform(data_comp)
        case 'mean':
            b_scaled = data_base - np.mean(data_base, axis=0)
            c_scaled = data_comp - np.mean(data_comp, axis=0)
        case _:
            raise ValueError("Invalid scaling keyword")

    b_pca = PCA(n_components=num_components)
    c_pca = PCA(n_components=num_components)

    b_proj = b_pca.fit_transform(b_scaled)
    c_proj = c_pca.fit_transform(c_scaled)

    var_difference = factor * sum(abs(b_pca.explained_variance_ratio_- c_pca.explained_variance_ratio_))

    # len_b = np.sqrt(b_pca.components_[0].dot(b_pca.components_[0]))
    # len_c = np.sqrt(c_pca.components_[0].dot(c_pca.components_[0]))

    angle_diff = min([np.arccos(_bound(b_pca.components_[0] @ (s*c_pca.components_[0]),(-1,1))) for s in [1,-1]])#/(len_r*len_f)

    results = {'exp_var_diff': var_difference, 'comp_angle_diff': (angle_diff*2)/np.pi}
    return results, b_proj, c_proj


def AAD(X: DataFrame, selected_features: List[str]) -> float:
    """Function for calculating the average angle difference of the selected features

    Args:
        X (DataFrame): The full dataset
        selected_features (List[str]): The list of selected features
    
    Returns:
        float: The average angle difference score for the selected features
    
    Example:
    >>> import pandas as pd
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> X = pd.DataFrame(iris.data, columns=iris.feature_names)
    >>> selected_features = ["sepal length (cm)", "sepal width (cm)"]
    >>> AAD(X, selected_features) # doctest: +ELLIPSIS
    0.32...
    
    """
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

if __name__ == "__main__":
    import doctest
    doctest.testmod()
