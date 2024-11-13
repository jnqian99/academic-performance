# Student Academic Outcome Prediction

## Overview

This project aims to predict student academic outcomes, focusing on identifying at-risk students who may drop out. Using a dataset from the UCI Machine Learning Repository, we apply data preprocessing, exploratory data analysis (EDA), clustering, outlier detection, feature selection, and classification to understand the factors influencing student success. Our findings can inform targeted intervention strategies for educational institutions to improve retention rates.

## Project Structure

This repository contains the following notebooks:
- **1_preprocess.ipynb**: Data preprocessing.
- **2_eda.ipynb**: Exploratory Data Analysis.
- **3_cluster.ipynb**: Clustering analysis.
- **4_outlier.ipynb**: Outlier detection.
- **5_feature_selection.ipynb**: Feature selection.
- **knn_classify.ipynb, rf_classify.ipynb, svm_classify.ipynb**: Classification using k-NN, Random Forest, and SVM.

Each section in this README details the tasks, results, visualizations, and code references.

---

## 1. Data Preprocessing

Preprocessing ensures data quality for reliable analysis and modeling. Key steps included:
- **Missing Value Handling**: Imputation of numerical values with medians and filling categorical values with placeholders.
- **Encoding**: Categorical features were label-encoded (low cardinality) or one-hot encoded (high cardinality, e.g., Nationality).
- **Normalization**: Numerical academic features were normalized to facilitate clustering and improve model performance.

> Refer to [1_preprocess.ipynb](./1_preprocess.ipynb) for code and results.

## 2. Exploratory Data Analysis (EDA)

EDA provided insight into feature distributions and relationships:
- **Summary Statistics**: Distribution analysis revealed skewness in grades and enrollment status, suggesting early intervention opportunities.
- **Correlation Heatmap**: Academic features, such as grades and credited units, showed strong correlations with student outcomes.
- **Cluster Patterns**: Initial analysis revealed separable groups based on academic performance, hinting at distinct student types.

> See [2_eda.ipynb](./2_eda.ipynb) for visualizations and analysis.

## 3. Clustering

Unsupervised clustering explored potential groupings of students:
- **KMeans Clustering**: The result indicated two clusters, though these did not align perfectly with the target classes.
- **Agglomerative Clustering**: Validated KMeans findings, supporting early indications of distinct student subgroups.
- **PCA Visualization**: 2D and 3D visualizations using PCA showed partial separation, suggesting clustering alone may not capture dropout risk fully.

> Refer to [3_cluster.ipynb](./3_cluster.ipynb) for detailed clustering steps and visuals.

## 4. Outlier Detection

Outlier detection was applied to highlight atypical students:
- **Isolation Forest, Local Outlier Factor, and Elliptic Envelope**: These methods identified students with unusual academic patterns. Multiple methods helped ensure robustness.
- **Visualization**: PCA-based scatter plots showed the spread and positioning of outliers.

> Check [4_outlier.ipynb](./4_outlier.ipynb) for the outlier detection process and visualizations.

## 5. Feature Selection

Feature selection was applied to improve model efficiency and accuracy:
- **Recursive Feature Elimination (RFE)**: Used to rank features, helping to retain those with the strongest predictive power.
- **Random Forest Feature Importance**: Provided insights into feature relevance, with academic performance metrics emerging as significant predictors.

> See [5_feature_selection.ipynb](./5_feature_selection.ipynb) for feature selection results.

## 6. Classification

Classification models were used to predict student outcomes:
- **k-Nearest Neighbors (k-NN)**: 
- **Support Vector Machine (SVM)**: 
- **Random Forest**: 

> See the classification notebooks ([knn_classify.ipynb](./knn_classify.ipynb), [rf_classify.ipynb](./rf_classify.ipynb), and [svm_classify.ipynb](./svm_classify.ipynb)) for detailed implementation and results.

## 7. Hyperparameter Tuning


---

## Conclusion and Future Work

### Key Findings


### Challenges and Limitations


### Future Enhancements


---

This README serves as a comprehensive report for the project, with links to the code in each notebook for further exploration.