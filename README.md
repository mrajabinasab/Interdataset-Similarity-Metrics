# Inter-Dataset Similarity Metric Based on PCA

This document presents the implementation of two novel metrics for measuring inter-dataset similarity based on PCA. These metrics are proposed in our paper, "Metrics for Inter-Dataset Similarity with Example Applications in Synthetic Data and Feature Selection Evaluation". This paper has been accepted to the 2025 SIAM International Conference on Data Mining (SDM). 

Jupyter notebooks are provided for the experiments presented in the paper. You can run the code to reproduce the results.

| Notebook | Description |
| --- | --- |
| [Pre-Investigation](https://github.com/mrajabinasab/Interdataset-Similarity-Metrics/blob/main/Theoretical_Preinvestigation.ipynb) | Investigation of general properties of the new metrics |
| [Use Case 1](https://github.com/mrajabinasab/Interdataset-Similarity-Metrics/blob/main/Synthetic_Data_Evaluation.ipynb) | Examples from the paper for evaluation of synthetic tabular data |
| - [Figure 3](https://github.com/notna07/ctgan-with-checkpoints/blob/main/gen_model_training_behaviour.ipynb) | Codebook in different repository due to licensing |
| - [Figure 4](https://github.com/schneiderkamplab/syntheval-model-benchmark-example/blob/main/metric_correlations.ipynb) | Codebook in different repository to avoid duplicate information |
| [Use Case 2](https://github.com/mrajabinasab/Interdataset-Similarity-Metrics/blob/main/Feature_Selection_Evaluation.ipynb) | Experiments from the paper on feature selection evaluation |

## Installation

You can install the package using pip:

```bash
pip install pcametric
```

## Usage

Below is an example of how to use the metrics:

```python
from pcametric import PCAMetric
import pandas as pd 

# Loading the datasets
df1 = pd.read_csv('df1.csv')
df2 = pd.read_csv('df2.csv')

# Setting parameters
num_components = 1
normalization = "precise"
preprocess = "std"

# Calculate the values of the two metrics, namely Difference in Explained Variance and Angle Difference
result, _, _ = PCAMetric(df1, df2, num_components, normalization, preprocess)
edv, ad = result['exp_var_diff'], result['comp_angle_diff']
```

The Average Angle Difference (AAD) metric is also implemented and can be used as a model-agnostic approach for evaluating the performance of feature selection:

```python
from pcametric import AAD
import pandas as pd 

# Loading the dataset
df = pd.read_csv('df.csv')

#Index of selected features
selected_features = [2, 5, 11, 17, 22, 31, 40] 

# Calculate AAD
aad = AAD(df, selected_features)
```

It is noteworthy that for all the metrics above, a lower value indicates greater similarity to the actual data.

### Citation

If you use our metrics in your research, please cite the original paper:

```
@inbook{doi:10.1137/1.9781611978520.57,
author = {Muhammad Rajabinasab and Anton Lautrup and Arthur Zimek},
title = {Metrics for Inter-Dataset Similarity with Example Applications in Synthetic Data and Feature Selection Evaluation},
booktitle = {Proceedings of the 2025 SIAM International Conference on Data Mining (SDM)},
chapter = {},
pages = {527-537},
doi = {10.1137/1.9781611978520.57},
URL = {https://epubs.siam.org/doi/abs/10.1137/1.9781611978520.57},
eprint = {https://epubs.siam.org/doi/pdf/10.1137/1.9781611978520.57},
}
```
