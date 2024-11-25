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
- ### KMeans Clustering: 
    #### Methodology and Results
    To determine the optimal number of clusters (`k`), we calculated the **Silhouette Score** for values of `k` ranging from 2 to 10. As shown in the plot below, the highest Silhouette Score was achieved at `k=3`. However, since the **Silhouette Score** for `k=2` and `k=3` are very close  (less than 0.03 difference), **domain knowledge** and the specific problem context should play a significant role in selecting the optimal number of clusters. In this case, since the problem is binary (e.g., “dropout” vs. “not dropout”), using `k=2` might align better with the data’s underlying structure.
    
    For k=4 and beyond, the Silhouette Score dropped significantly to some very low value (~0.09) suggests that the clusters become less meaningful, possibly due to splitting existing clusters or introducing noise. Overall, the relatively low scores (maximum ~0.25) suggest that the dataset does not form highly distinct clusters, which might result from overlapping features or noise.

    **We chose k = 2 as the optimal number of clusters of Kmeans.**

    ![Silhouette Score](results/clustering_silhouette_kmeans.png)

    After selecting `k=2`, we applied K-Means clustering to the dataset. To visualize the clusters, we reduced the data to **3** dimensions using **Principal Component Analysis (PCA)**. The resulting clusters are displayed in the 3D scatter plot below. Each color represents a distinct cluster.

    ![K-Means Clustering](results/clustering_3d_kmeans.png)

    The 3D scatter plot of KMeans clusters in the PCA-reduced space shows three distinct groups:
	-	One cluster is densely packed, while the other two are more dispersed.
	-	The separation between clusters is visible, but some overlap exists, particularly between two clusters.
	-	This indicates that while the algorithm can group the data meaningfully, the feature space may not fully separate the underlying groups.

    We also compared the clusters to the actual target labels (dropout, not dropout) by coloring the points based on their true categories. This revealed some overlap between the clusters and target labels, as shown below:

    ![Target vs. Clusters](results/clustering_3d_target_kmeans.png)

    #### Insights
    - The results suggest that K-Means with `k=2` groups students into clusters, likely representing at-risk students, and average performers.
    - The overlap between clusters and target labels suggests the clusters capture general groupings but do not well align with the true outcomes. This indicates potential for further refinement, such as using additional clustering methods or improving feature engineering.

- ### Hierarchical Clustering: 

    #### Methodology and Results

    To determine the optimal number of clusters (`k`), we calculated the **Silhouette Score** for values of `k` ranging from 2 to 10 using Agglomerative Clustering (Hierarchical Clustering). The plot below displays the **Silhouette Scores** for different values of `k`:

    ![Silhouette Score](results/clustering_silhouette_hiera.png)

    - The highest **Silhouette Score** of **0.23** was observed at **k=4**.
    - Scores for **k=2** and **k=3** were also relatively high, which aligns with the binary nature of the dataset (dropout vs. not dropout).
    - Beyond `k=4`, the scores drop significantly, indicating that higher cluster values might introduce noise or split meaningful groupings.
    - Overall, the relatively low scores (maximum ~0.25) suggest that the dataset does not form highly distinct clusters, which might result from overlapping features or noise.

    After selecting **k=4**, Hierarchical Clustering was applied to the dataset. The data was reduced to **3 dimensions** using **Principal Component Analysis (PCA)** for visualization. Two 3D scatter plots were generated to display the clusters from different viewpoints:

    ![Hierarchical Clustering](results/clustering_3d_hiera.png)

    #### Insights

    The 3D scatter plots reveal the following:
    1. Four distinct clusters are visible, with varying densities and overlaps.
    2. Overlaps between clusters suggest some complexity in the dataset that hierarchical clustering struggles to fully separate.

    To analyze the relationship between clusters and the actual target labels, the clusters were visualized with true labels in a separate plot:

    ![Clusters and True Labels](results/clustering_3d_target_agg.png)

    #### Observations
    1. Comparing these clusters with true labels (dropout vs. not dropout) reveals that while some alignment exists, significant overlaps remain.
    3. Overlaps between clusters and true labels suggest that hierarchical clustering captures general groupings but does not separate the classes well.

- ### DBSCAN:

    #### -> Methodology and Results

    To determine the optimal `eps` for DBSCAN, a **k-distance graph** was used. The sorted distances to the 5th nearest neighbor were plotted, and the elbow point was identified based on the curvature of the graph. The resulting optimal `eps` value was found to be **4.35**.

    ![K-Distance Graph](results/clustering_kdis_dbscan.png)

    #### ->Evaluation of DBSCAN with Different `eps` Values

    DBSCAN was evaluated with various `eps` values ranging from **4.0** to **5.5** (based on the best eps get from k-distances). For each `eps`, the following metrics were calculated:
    - **Number of clusters** formed (excluding noise).
    - **Number of noise points**.
    - **Silhouette Score** to evaluate clustering quality.

    | `eps` | Number of Clusters | Noise Points | Silhouette Score |
    |-------|--------------------|--------------|------------------|
    | 4.0   | 7                  | 300          | 0.2667           |
    | 4.35  | 5                  | 169          | 0.2959           |
    | 5.0   | 3                  | 53           | 0.3454           |
    | 5.5   | 2                  | 20           | 0.5642           |

    ![Silhouette Score vs eps](results/clustering_silhouette_DBSCAN.png)

    - The highest **Silhouette Score** of **0.5642** was achieved at `eps=5.5`, but this configuration resulted in only 2 clusters, which may oversimplify the data.
    - At **eps=4.35**, a moderate Silhouette Score of **0.2959** was achieved with **5 clusters**, which balances granularity and clustering quality.

    #### Visualizing Clusters

    Using **eps=4.35**, DBSCAN was applied to the dataset. Two 3D scatter plots were generated to visualize the clusters from different perspectives, with noise points (cluster `-1`) clearly marked as 'x':

    ![DBSCAN Clustering](results/clustering_3d_dbscan.png)

    The relationship between DBSCAN clusters and true target labels (dropout vs. not dropout) was also visualized. Each cluster was annotated with both its ID and the corresponding target label:

    ![Clusters and True Labels](results/clustering_3d_target_dbscan_views_with_noise.png)

    #### Observations

    1. **Dominance of One Main Cluster**:
    - While DBSCAN identified 5 clusters, **most points are concentrated in one main cluster**, with the remaining clusters sparsely distributed.
    - This suggests that DBSCAN did not effectively separate the dataset into meaningful subgroups.

    2. **Silhouette Score**:
    - The Silhouette Score for `eps=4.35` was **0.2959**, which indicates poor clustering performance.
    - Low Silhouette Scores suggest that the clusters overlap significantly or are not well-separated.

    3. **Comparison with True Labels**:
    - When comparing the clusters with the true labels (dropout vs. not dropout), the main cluster contains a **mix of both dropout and non-dropout points**.
    - This lack of separation in the main cluster suggests that DBSCAN failed to distinguish between the two classes effectively.

### Conclusion of Clustering

The clustering analysis using KMeans, Hierarchical Clustering, and DBSCAN revealed insights into the dataset's structure, but all methods faced challenges due to overlapping features and noise, as evidenced by relatively low Silhouette Scores.

- **Clustering Performance**:
  - Across all methods, the Silhouette Scores remained below **0.3**, indicating weak separation of clusters. 
  - KMeans and Hierarchical Clustering performed similarly, with moderate separation of clusters but noticeable overlaps with the true target labels (dropout vs. not dropout).
  - DBSCAN struggled the most, as it grouped most data points into a single dominant cluster and did not effectively separate dropout and non-dropout points.

- **Noise and Overlap**:
  - DBSCAN identified noise points effectively but failed to cluster the remaining data meaningfully.
  - KMeans and Hierarchical Clustering produced clusters that overlapped significantly, suggesting that the features do not sufficiently separate the two target groups.

- **Comparison**:
  - **KMeans** is the most interpretable method and aligned best with the binary nature of the dataset (dropout vs. not dropout) when `k=2` was used.
  - **Hierarchical Clustering** provided insights into potential subgroup relationships but did not outperform KMeans.
  - **DBSCAN**, while effective in identifying noise, struggled with meaningful clustering for this dataset.

#### Final Takeaways
- The results indicate that the dataset is not inherently well-suited for clustering due to overlapping features and noise. 
- The dataset's high dimensionality and categorical features may hinder the effectiveness of clustering. 

> Refer to [3_cluster.ipynb](./3_cluster.ipynb) for detailed clustering steps and visuals.


## 4. Outlier Detection


Outlier detection was applied to highlight atypical students:
- **Isolation Forest, Local Outlier Factor, and Elliptic Envelope**: These methods identified students with unusual academic patterns. Multiple methods helped ensure robustness.
- **Visualization**: PCA-based scatter plots showed the spread and positioning of outliers.


> Check [4_outlier.ipynb](./4_outlier.ipynb) for the outlier detection process and visualizations.


## 5. Feature Selection

### Methodology

To address the high dimensionality of the dataset and improve model performance, feature selection techniques were applied. The methods used include:

1. **Recursive Feature Elimination (RFE)**:
   - This method iteratively removes less important features based on their impact on model performance.
   - The number of features selected was varied from 1 to the total number of features, and the **accuracy** was evaluated for each selection.

2. **Mutual Information**:
   - This method quantifies the dependency between each feature and the target variable.
   - Features with higher mutual information scores are considered more relevant to the classification task.

### Results and Analysis

1. **Recursive Feature Elimination (RFE)**:
   - The accuracy plot below shows the model performance as the number of selected features increases:
   
   ![RFE Accuracy](results/feature_accuracy.png)
   
   - Key observations:
     - The accuracy increased significantly as the number of selected features increased up to **9 features**, reaching a maximum of **0.87**.
     - After this point, the accuracy plateaued, suggesting that additional features beyond the top 9 provided diminishing returns.
     - The model with **12 features** was tested on the test set, achieving an accuracy of **84.52%**.

2. **Mutual Information**:
   - The plot below ranks the features based on their mutual information scores:
   
   ![Mutual Information](results/feature_mutual_information.png)
   
   - Key observations:
     - Features such as **"Curricular units 2nd sem (approved)"**, **"Curricular units 2nd sem (grade)"**, and **"Curricular units 1st sem (grade)"** were the most informative, indicating their strong correlation with the target variable.
     - Features with low mutual information scores (e.g., **"Inflation rate"**, **"Educational special needs"**) were less relevant.

   - Using the top **9 features** ranked by mutual information, the model achieved an accuracy of **0.87**, similar to the RFE results.

3. **Comparison of Model Performance**:
   - The table below compares the accuracy with and without feature selection:
   
   | Method                  | Number of Features | Accuracy (%) |
   |-------------------------|--------------------|--------------|
   | Without Feature Selection | 34                 | 84.52        |
   | RFE (Top 12 Features)    | 12                 | 84.52        |
   | Mutual Information (Top 9 Features) | 9          | 87.00        |
   
   - Feature selection did not significantly improve accuracy but reduced dimensionality.

### Insights

- Feature selection identified the most relevant features for the classification task, reducing dimensionality while maintaining or slightly improving model performance.
- The **top features** (e.g., **"Curricular units"**, **"Tuition fees"**, and **"Application order"**) are directly related to student progress and enrollment, aligning with the domain knowledge of student dropout prediction.
- Despite overlapping results, mutual information provided a clearer ranking of features, which can guide future model refinement.


> See [5_feature_selection.ipynb](./5_feature_selection.ipynb) for feature selection results.


## 6. Classification
Because our goal for this project is to identify and to help struggling students, we want the False Negative counts to be as small as possible. That means we do not want struggling students to be identified as not struggling and eventually drop out due to our classification error and not helping them. Therefore, we use the recall as the primary score but would also consider accuracy. This is because if we only consider recall then it is best to identify every target as True and then help every student which is probably not possible due to limited teaching and aid resources.
During hyperparameter searching for the classifiers, we generally use roc-auc instead of accuracy as the scoring function. This can improve recall while keeping the accuracy in a relatively high range. We used the GridSearchCV function which performed 5-fold cross validation on the training data for the hyperparameter search.
The prediction is generated as a probability which can be determined by a probability threshold value whether the sample belongs to the positive or negative class. This threshold value constitutes another hyperparameter that should be searched before predicting the testing data. During this hyperparameter search, 80% of the training data was used for training the model and the rest of the training data was used as the validation data.


- **Support Vector Machine (SVM)**:
For SVM, the class_weight parameter is chosen to be “balanced” to increase the class weight for the positive class (Dropout) based on its corresponding sample frequency relative to the negative class (Enrolled or Graduate). 
Hyperparameter search was performed based on scoring for “roc-auc” and searched on parameter C, gamma, and kernel. The result roc_auc = 0.9312 when C = 200, gamma = 0.0001, kernel = rbf. 
Hyperparameter search was also performed on the decision probability threshold. A threshold value of 0.12 is chosen to achieve a higher recall score while keeping the accuracy score relatively reasonable.
Ten fold cross validation was performed on the training data with accuracy mean = 0.8705 and accuracy std = 0.0113 showing that the model is consistent with different training samples.
The fitted SVM model was used to predict the test data to achieve recall = 0.8132 and accuracy = 0.8497. When the decision probability threshold is set to be 0.12, it achieved a recall value of 0.9177 and accuracy of 0.7853. That means we can identify and help over 90% of the struggling students while not wasting too much aiding resources on non-struggling students.
The prediction probability values were then used to plot the roc curve using the built-in roc_curve function by setting different thresholds to emulate different models. It achieved a resulting roc-auc of 0.9134.


> Check [svm_classify.ipynb](./notebooks/6_classification/svm_classify.ipynb) for details


- **Random Forest**:
Similar to SVM, a “balanced” class_weight is chosen for the classifier to avoid skewing in sample classes.
Hyperparameter search was performed based on scoring for “roc-auc” and searched on parameter n_estimators, max_depth, and criterion. The result roc_auc = 0.9194 on the training data when max_depth = 50, n_estimators = 300, criterion = entropy. 
Hyperparameter search was also performed on the decision probability threshold. A threshold value of 0.16 is chosen to achieve a higher recall score while keeping the accuracy score relatively reasonable.
Ten fold cross validation was performed on the training data with accuracy mean = 0.8765 and accuracy std = 0.0083 showing that the model is consistent with different training samples.
The fitted Random Forest model was used to predict the test data to achieve recall = 0.8955 and accuracy = 0.7548 with decision probability threshold set to be 0.16.
The prediction probability values were then used to plot the roc curve using the built-in roc_curve function by setting different thresholds to emulate different models. It achieved a resulting roc-auc of 0.9078.


> Check ([rf_classify.ipynb](./rf_classify.ipynb) for implementation and detailed results.


- **k-Nearest Neighbors (k-NN)**:
For k-NN, a hyperparameter search is performed on parameter n_neighbors, weights, and metric with roc-auc as the scoring function. The maximum roc-auc = 0.8952 is reached when n_neighbors = 77, metric = manhattan and weights = distance. 
A hyperparameter search for the probability threshold is then performed on the training data, and threshold = 0.15 was determined to be a balanced choice between recall and accuracy with emphasis on recall score.
We used 5-fold cross validation on the training data with resulting accuracy mean = 0.8174 and accuracy std = 0.0068. This showed that the model is consistent with different training data sets.
The fitted k-NN model was used to predict the test data to achieve recall = 0.8449 and accuracy = 0.7649 with decision probability threshold set to be 0.15.
The prediction probability values were then used to plot the roc curve using the built-in roc_curve function by setting different thresholds to emulate different models. It achieved a resulting roc-auc of 0.8763.


> Check ([knn_classify.ipynb](./knn_classify.ipynb) for implementation and detailed results.


## 7. Hyperparameter Tuning
See section Classification for related hyperparameter tuning details.
---


## Conclusion and Future Work


### Key Findings




### Challenges and Limitations




### Future Enhancements




---


This README serves as a comprehensive report for the project, with links to the code in each notebook for further exploration.

