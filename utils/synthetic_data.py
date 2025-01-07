# Description: File for using various generative models for synthetic data generation.
# Author: Anton D. Lautrup
# Date: 07-01-2025

import os

import numpy as np
import pandas as pd

from typing import List
from pandas import DataFrame

def _cleanup_files(file_names: List[str]) -> None:
    for file_name in file_names:
        if os.path.exists(file_name):
            os.remove(file_name)
    pass

from astropy.stats import biweight_midvariance
def add_noise_to_dataset(dataset: DataFrame, noise_level: float = 0.1, threshold: int = 3) -> DataFrame:
    """ Adds noise to a dataset consisting of numerical and categorical variables.
    
    Args:
        dataset (DataFrame): The dataset to add noise to.
        noise_level (float): The level of noise to add. Default is 0.1.
        threshold (int): The threshold to determine categorical variables. Default is 3.
        
    Returns:
        DataFrame: The dataset with added noise.

    Example:
    >>> import pandas as pd
    >>> data = {'A': [1, 2, 3, 4, 5], 'B': ['a', 'b', 'c', 'd', 'e'], 'C': [1.1, 2.2, 3.3, 4.4, 5.5]}
    >>> df = pd.DataFrame(data)
    >>> noisy_df = add_noise_to_dataset(df, noise_level=0.1)
    >>> noisy_df # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
       A  B    C
    ...
    """
    noisy_dataset = dataset.copy()
    
    for column in noisy_dataset.columns:
        if noisy_dataset[column].dtype == 'object' or len(noisy_dataset[column].unique()) <= threshold:
            continue
        elif (noisy_dataset[column].dtype == 'int64' or noisy_dataset[column].apply(float.is_integer).all()):
            continue         
        else:
            noise = biweight_midvariance(noisy_dataset[column])
            noisy_dataset[column] = noisy_dataset[column].apply(lambda x: x + noise_level*np.random.normal(0, noise) if pd.notnull(x) else x)
    
    return noisy_dataset

def independent_sampling(dataset: DataFrame) -> DataFrame:
    """ Function to sample each variable independently from the others in a dataset.
    
    Args:
        dataset (DataFrame): The dataset to sample from.
        
    Returns:
        DataFrame: The sampled dataset.

    Example:
    >>> import pandas as pd
    >>> data = {'A': [1, 2, 3, 4, 5], 'B': ['a', 'b', 'c', 'd', 'e'], 'C': [1.1, 2.2, 3.3, 4.4, 5.5]}
    >>> df = pd.DataFrame(data)
    >>> sampled_df = independent_sampling(df)
    >>> sampled_df # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
       A  B    C
    ...
    """
    sampled_dataset = pd.DataFrame()
    
    for column in dataset.columns:
        sampled_dataset[column] = dataset[column].sample(frac=1, replace=True).reset_index(drop=True)
    
    return sampled_dataset


from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
def generate_using_datasynthesizer(dataset: DataFrame) -> DataFrame:
    """ Function to generate synthetic data using the DataSynthesizer package.

    Args:
        dataset (DataFrame): The dataset to generate synthetic data from.
    
    Returns:
        DataFrame: The synthetic dataset.
    
    Example:
    >>> import pandas as pd
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> df = pd.DataFrame(iris.data, columns=iris.feature_names)
    >>> synthetic_df = generate_using_datasynthesizer(df) # doctest: +ELLIPSIS
    ================ Constructing Bayesian Network (BN) ================
    ...
    >>> isinstance(synthetic_df, pd.DataFrame)
    True
    """
    dataset.to_csv("datasynthesizer_temp.csv", index=False)

    description_file = "datasynthesizer_info.json"

    describer = DataDescriber(category_threshold=10)
    describer.describe_dataset_in_correlated_attribute_mode(dataset_file = 'datasynthesizer_temp.csv', 
                                                            epsilon=0, 
                                                            k=2,
                                                            attribute_to_is_categorical={}
                                                            )
    describer.save_dataset_description_to_file(description_file)

    generator = DataGenerator()

    num_to_generate = len(dataset)
    generator.generate_dataset_in_correlated_attribute_mode(num_to_generate, description_file)

    df_syn = generator.synthetic_dataset

    _cleanup_files([description_file, 'datasynthesizer_temp.csv'])

    return df_syn

from synthcity.plugins import Plugins
def generate_using_adsgan(dataset: DataFrame) -> DataFrame:
    """ Function to generate synthetic data using the ADS-GAN model.

    Args:
        dataset (DataFrame): The dataset to generate synthetic data from.
    
    Returns:
        DataFrame: The synthetic dataset.
    
    Example:
    >>> import pandas as pd
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> df = pd.DataFrame(iris.data, columns=iris.feature_names)
    >>> synthetic_df = generate_using_adsgan(df) # doctest: +ELLIPSIS
    """
    syn_model = Plugins().get("adsgan")
    syn_model.fit(dataset)

    num_to_generate = len(dataset)
    df_syn = syn_model.generate(count=num_to_generate).dataframe()

    return df_syn


if __name__ == "__main__":
    import doctest
    doctest.testmod()