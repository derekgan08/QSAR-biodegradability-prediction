# CPC251 - Machine Learning and Computational Intelligence Project: QSAR Biodegradability Prediction

## Problem Statements
The growing use of chemicals in various industries highlights the need for effective methods to predict their environmental impact, especially in terms of biodegradability. As biodegradation is a crucial mechanism for the removal of organic chemicals in natural systems, accurately predicting a chemical's biodegradability is essential for mitigating environmental risks and improving the design of more sustainable chemicals. 

This is particularly important for chemicals that enter aquatic environments, whether in large or small quantities. Estimating their biodegradability is necessary for assessing the full scope of their potential hazards. Therefore, developing reliable methods to quickly and accurately diagnose and analyze biodegradability is critical to ensuring safer chemical use and minimising environmental harm.

## Project Overview
This project aims to build an effective and reliable predictive model that is able to accurately predict the biodegradability of chemical compounds using QSAR (Quantitative Structure-Activity Relationship) models. This project employs four predictive modeling techniques: K-Nearest Neighbor (KNN), Decision Tree, Neural Network, and Logistic Regression. The goal is to compare the performance of these models based on accuracy, recall, precision, F1-score, and confusion matrix metrics.

The dataset used is a QSAR biodegradation dataset, which includes molecular descriptors for various chemical compounds, and their corresponding biodegradability class (RB or NRB).

## Dataset Description
The QSAR biodegradation dataset was built in the Milano Chemometrics and QSAR Research Group (Università degli Studi Milano-Bicocca, Milano, Italy). The research leading to these results has received funding from the European Community's Seventh Framework Programme [FP7/2007-2013] under Grant Agreement n. 238701 of Marie Curie ITN Environmental Chemoinformatics (ECO) project.

The data have been used to develop QSAR (Quantitative Structure Activity Relationships) models for the study of the relationships between chemical structure and biodegradation of molecules. Biodegradation experimental values of 1055 chemicals were collected from the webpage of the National Institute of Technology and Evaluation of Japan (NITE). Classification models were developed in order to discriminate ready (356) and not ready (699) biodegradable molecules by means of three different modelling methods: k Nearest Neighbours, Partial Least Squares Discriminant Analysis and Support Vector Machines. Details on attributes (molecular descriptors) selected in each model can be found in the quoted reference: Mansouri, K., Ringsted, T., Ballabio, D., Todeschini, R., Consonni, V. (2013). Quantitative Structure - Activity Relationship models for ready biodegradability of chemicals. Journal of Chemical Information and Modeling, 53, 867-878.

<p align="center">
<img src="readme-assets/01 Sample of the QSAR Biodegradation Dataset.png" alt="Sample of the QSAR Biodegradation Dataset"/>
<br/>
<i>Table 1: Sample of the QSAR Biodegradation Dataset</i>
</p>

The data set contains values for 41 attributes (molecular descriptors) to classify 1055 chemicals into 2 classes (ready and not ready biodegradable). The attribute information of 41 molecular descriptors and 1 experimental class is as below:

1. `SpMax_L`: Leading eigenvalue from Laplace matrix
2. `J_Dz(e)`: Balaban-like index from Barysz matrix weighted by Sanderson electronegativity
3. `nHM`: Number of heavy atoms
4. `F01[N-N]`: Frequency of N-N at topological distance 1
5. `F04[C-N]`: Frequency of C-N at topological distance 4
6. `NssssC`: Number of atoms of type ssssC
7. `nCb-`: Number of substituted benzene C(sp2)
8. `C%`: Percentage of C atoms
9. `nCp`: Number of terminal primary C(sp3)
10. `nO`: Number of oxygen atoms
11. `F03[C-N]`: Frequency of C-N at topological distance 3
12. `SdssC`: Sum of dssC E-states
13. `HyWi_B(m)`: Hyper-Wiener-like index (log function) from Burden matrix weighted by mass
14. `LOC`: Lopping centric index
15. `SM6_L`: Spectral moment of order 6 from Laplace matrix
16. `F03[C-O]`: Frequency of C - O at topological distance 3
17. `Me`: Mean atomic Sanderson electronegativity (scaled on Carbon atom)
18. `Mi`: Mean first ionization potential (scaled on Carbon atom)
19. `nN-N`: Number of N hydrazines
20. `nArNO2`: Number of nitro groups (aromatic)
21. `nCRX3`: Number of CRX3
22. `SpPosA_B(p)`: Normalized spectral positive sum from Burden matrix weighted by polarizability
23. `nCIR`: Number of circuits
24. `B01[C-Br]`: Presence/absence of C - Br at topological distance 1
25. `B03[C-Cl]`: Presence/absence of C - Cl at topological distance 3
26. `N-073`: Ar2NH / Ar3N / Ar2N-Al / R..N..R
27. `SpMax_A`: Leading eigenvalue from adjacency matrix (Lovasz-Pelikan index)
28. `Psi_i_1d`: Intrinsic state pseudoconnectivity index - type 1d
29. `B04[C-Br]`: Presence/absence of C - Br at topological distance 4
30. `SdO`: Sum of dO E-states
31. `TI2_L`: Second Mohar index from Laplace matrix
32. `nCrt`: Number of ring tertiary C(sp3)
33. `C-026`: R--CX--R
34. `F02[C-N]`: Frequency of C - N at topological distance 2
35. `nHDon`: Number of donor atoms for H-bonds (N and O)
36. `SpMax_B(m)`: Leading eigenvalue from Burden matrix weighted by mass
37. `Psi_i_A`: Intrinsic state pseudoconnectivity index - type S average
38. `nN`: Number of Nitrogen atoms
39. `SM6_B(m)`: Spectral moment of order 6 from Burden matrix weighted by mass
40. `nArCOOR`: Number of esters (aromatic)
41. `nX`: Number of halogen atoms
42. `experimental class`: ready biodegradable (RB) and not ready biodegradable (NRB)

## Data Analysis
### K-Nearest Neighbors (KNN) and Decision Tree
For K-Nearest Neighbors (KNN) and Decision Tree, chi squared method is used as the feature selection method because chi squared works well on a categorical dataset. The dataset response is a binary classification, which has classes of 'RB' and 'NRB' as its values. This is equivalent to true or false value in binary classification. So, chi squared is the best features selection method for our dataset for both prescriptive models, K-Nearest Neighbours and Decision Tree.

To plot the graph, ten features out of 41 features and one response were chosen. Because the top ten features have the lowest values of all the features, this is the case. Each feature with the lowest value has the top ten highest correlation with the set of data. These are the top ten features, as determined by the chi square method, with the lowest possible scores:

<p align="center">
<img src="readme-assets/02 The Selected Features for KNN and Decision Tree.png" alt="The Selected Features"/>
<br/>
<i>Table 2: The Selected Features for KNN and Decision Tree</i>
</p>

### Neural Network and Logistic Regression

For Neural Network and Logistic Regression, 7he feature selection method is ANOVA (Analysis of Variance). ANOVA is the best fit because the input variable is numerical and the target output is a categorical dataset with classes of 'RB' and 'NRB'. The dataset response is a binary classification equivalent to a true or false value. As a result, the ANOVA method is the best feature selection method for the QSAR dataset for both the prescriptive Neural Network model and the logistic model.

When choosing features, filtering techniques were used. The statistical method is being used to determine the correlation between each input variable and the target variable. It determines the strength of the correlation between the feature and the target response using the QSAR dataset to compute the probability (p) value and score. The lower p values obtained, the stronger the relationship was with the response. Lower p values were obtained the stronger the relationship was with the response. The relationship between the response and score increases as the score rises. Since the p-value for each feature is less than 0.05, all features were chosen. The list of features with p-value and score is shown below.

<p align="center">
<img src="readme-assets/12 The Selected Features for Neural Network and Logistic Regression.png" alt="The Selected Features"/>
<br/>
<i>Table 3: The Selected Features for Neural Network and Logistic Regression</i>
</p>

## Comparisons of the Machine Learning Models
### K-Nearest Neighbors (KNN) vs Decision Tree
<p align="center">
<img src="readme-assets/03 Comparisons Between K-Nearest Neighbors (KNN) and Decision Tree.png" alt="Comparisons Between K-Nearest Neighbors (KNN) and Decision Tree"/>
<br/>
<i>Table 4: Comparisons Between K-Nearest Neighbors (KNN) and Decision Tree</i>
</p>

### Neural Network vs Logistic Regression
<p align="center">
<img src="readme-assets/13 Comparisons Between Neural Network and Logistic Regression.png" alt="Comparisons Between K-Nearest Neighbors (KNN) and Decision Tree"/>
<br/>
<i>Table 5: Comparisons Between Neural Network and Logistic Regression</i>
</p>

## Data Modeling
For K-Nearest Neighbors and Decision Tree, the ratio split is 60% training 20% validation and 20% testing. The parameter for both predictive models is shown in Table 4.

<p align="center">
<img src="readme-assets/04 Parameter of the KNN and Decision Tree Predictive Models.png" alt="Parameter of the KNN and Decision Tree Predictive Models"/>
<br/>
<i>Table 6: Parameter of the KNN and Decision Tree Predictive Models</i>
</p>

### K-Nearest Neighbors (KNN)
For K-Nearest Neighbors model, we first need to find the best k value to train the model. We use the validation testing method to determine the optimized value of k. The model will be trained against the validation set until it finds the k value with the highest score.

<p align="center">
<img src="readme-assets/05 Result of The Best Value K.png" alt="Result of The Best Value K"/>
<br/>
<i>Figure 1: Result of The Best Value K</i>
</p>

<p align="center">
<img src="readme-assets/06 Analysis for the KNN Model Result.png" alt="Analysis for the KNN Model Result"/>
<br/>
<i>Figure 2: Analysis for the KNN Model Result</i>
</p>

<p align="center">
<img src="readme-assets/07 Decision Boundary of the Perceptron Classifier.png" alt="Decision Boundary of the Perceptron Classifier"/>
<br/>
<i>Figure 3: Decision Boundary of the Perceptron Classifier</i>
</p>

### Decision Tree
This is the report of the Decision Tree accuracy, precision, recall, f1-score, support and confusion matrix:

<p align="center">
<img src="readme-assets/08 Analysis for Decision Tree Model Result.png" alt="Analysis for Decision Tree Model Result"/>
<br/>
<i>Figure 4: Analysis for Decision Tree Model Result</i>
</p>

The diagram below shows a decision tree model with a max depth of 11.

<p align="center">
<img src="readme-assets/09 Decision Tree Model Diagram.png" alt="Decision Tree Model Diagram"/>
<br/>
<i>Figure 5: Decision Tree Model Diagram</i>
</p>

<p align="center">
<img src="readme-assets/10 Decision Boundary of the Decision Tree Classifier.png" alt="Decision Boundary of the Decision Tree Classifier"/>
<br/>
<i>Figure 6: Decision Boundary of the Decision Tree Classifier</i>
</p>

For Neural Network and Logistic Regression, ratio split is 80% training and 20% testing. The parameter for both predictive models is shown in Table 4.

<p align="center">
<img src="readme-assets/14 Parameter of the Neural Network and Logistic Regression Predictive Models.png" alt="Parameter of the Neural Network and Logistic Regression Predictive Models"/>
<br/>
<i>Table 7: Parameter of the Neural Network and Logistic Regression Predictive Models</i>
</p>

### Neural Network
Finding the optimal hyperparameters for the Neural network model is the first step in determining how the model will perform. The grid search method is the optimization technique used. This model's hyperparameter concentrates on the batch sizes, epochs, optimizer algorithm, and initialization mode that work best with this model's optimization. An optimizer named Adam has been used in this model.

<p align="center">
<img src="readme-assets/15 Neural Network - Result of the Best-Optimized Parameter.png" alt="PNeural Network - Result of the Best-Optimized Parameter"/>
<br/>
<i>Figure 7: Neural Network - Result of the Best-Optimized Parameter</i>
</p>

<p align="center">
<img src="readme-assets/16 Neural Network - Accuracy of Training and Test Dataset.png" alt="Neural Network - Accuracy of Training and Test Dataset"/>
<br/>
<i>Figure 8: Neural Network - Accuracy of Training and Test Dataset</i>
</p>

<p align="center">
<img src="readme-assets/17 Neural Network - Accuracy, Confusion Matrix and Classification Report.png" alt="Neural Network - Accuracy, Confusion Matrix and Classification Report"/>
<br/>
<i>Figure 9: Neural Network - Accuracy, Confusion Matrix and Classification Report</i>
</p>

<p align="center">
<img src="readme-assets/18 Accuracy Curve for the Neural Network Model Over Epochs.png" alt="Accuracy Curve for the Neural Network Model Over Epochs"/>
<br/>
<i>Figure 10: Accuracy Curve for the Neural Network Model Over Epochs</i>
</p>

<p align="center">
<img src="readme-assets/19 Loss Curve for the Neural Network Model Over Epochs.png" alt="Loss Curve for the Neural Network Model Over Epochs.png"/>
<br/>
<i>Figure 11: Loss Curve for the Neural Network Model Over Epochs</i>
</p>

### Logistic Regression
Grid Search CV was used to build a logistic regression model. This is the report of the Logistic Regression accuracy, precision, recall, f1-score, support and confusion matrix:

<p align="center">
<img src="readme-assets/20 Logistic Regression - Accuracy, Confusion Matrix and Classification Report.png" alt="Logistic Regression - Accuracy, Confusion Matrix and Classification Report"/>
<br/>
<i>Figure 12: Logistic Regression - Accuracy, Confusion Matrix and Classification Report</i>
</p>

## Insights

<p align="center">
<img src="readme-assets/21 Hierarchy of Metrics from Raw measurements or Labeled Data to F1-Score.png" alt="Hierarchy of Metrics from Raw measurements or Labeled Data to F1-Score"/>
<br/>
<i>Figure 13: Hierarchy of Metrics from Raw measurements or Labeled Data to F1-Score</i>
</p>

Precision is how precise or accurate the model is out of those predicted positive and how many of them are actual positive. Precision is a good measure to determine, when the costs of False Positive is high and we want to minimize false positives.

Recall calculates how many of the Actual Positives our model capture through labelling it as positive (True Positive). It shall be the model metric we use to select our best model when there is a high cost associated with False Negative and we want to minimize the chance of missing positive cases (predicting false negatives).

Accuracy is a good measure if we have quite balanced datasets and are interested in all types of outputs equally. It is largely contributed by a large number of True Negatives which in most scenarios, we do not focus on much.

F1 score is needed when we want to seek a balance between Precision and Recall. It is a better measure to use if the datasets are imbalanced and there is an uneven class distribution (large number of Actual Negatives).

In simple, a model is considered to be good if it gives high accuracy scores in production or test data or is able to generalise well on an unseen data. In our opinion, accuracy greater than 70% is considered as a great model performance.

## Discussion

### KNN and Decision Tree

<p align="center">
<img src="readme-assets/11 KNN and Decision Tree Discussion.png" alt="KNN and Decision Tree Discussion"/>
<br/>
<i>Figure 14: KNN and Decision Tree Discussion</i>
</p>

Based on the results of each model, we can observe that the accuracy score of the K-Nearest Neighbor model is higher compared to that of the Decision Tree. This can be concluded as the accuracy of the K-Nearest Neighbor model is more accurate and closer to the value one. Both models resulted in the same value of recall, precision, and F1-score, K-Nearest Neighbor. Therefore, we can conclude that the K-Nearest Neighbor algorithm is the better predictive model than Decision Tree to be used on the QSAR Biodegradation dataset.

### Neural Network and Logistic Regression

<p align="center">
<img src="readme-assets/17 Neural Network - Accuracy, Confusion Matrix and Classification Report.png" alt="Neural Network - Accuracy, Confusion Matrix and Classification Report"/>
<br/>
<i>Figure 15: Neural Network - Accuracy, Confusion Matrix and Classification Report</i>
</p>

<p align="center">
<img src="readme-assets/20 Logistic Regression - Accuracy, Confusion Matrix and Classification Report.png" alt="Logistic Regression - Accuracy, Confusion Matrix and Classification Report"/>
<br/>
<i>Figure 16: Logistic Regression - Accuracy, Confusion Matrix and Classification Report</i>
</p>

For both models, they have the same accuracy value and almost the same Precision, Recall and f1-score. Hence, we can conclude that both models have the same performance. However, we think that Logistic Regression is better than Neural Network because:

1. Although Neural Network can find more interesting patterns in the data, which can lead to better performance, it can be more complex and harder to build and maintain.

2. Logistic Regression has significantly lower training time and cost than Neural Network.

3. Logistic Regression has significantly lower inference time than Neural Network to run the model and making predictions.