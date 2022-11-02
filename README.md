# Dimensionality Reduction
Dimensionality Reduction  : LLE and LIM , Classifier : KNN, RF, MLP, RBF

Implementing dimensionality reduction\
Elhamsadat Hejazi(https://www.linkedin.com/in/elham-hejazi)

## Overal Work

## Introduction

Real-world data for many machine-learning or data-mining projects are usually high-dimensional or consist of a large number of features which is hard to be managed. To avoid overfitting, wasting time and storage for saving data, and to improve the performance of the machine-learning model, there are some algorithms that scientists use to achieve low-dimensional data out of the original dataset provided that it retains most of the important attributes, called intrinsic, of the main dataset. This transformation is called Dimensionality Reduction (DR). The DR algorithms listed in the project are Probabilistic Principal Component Analysis (PPCA), Landmark ISOMAP [LIM], and Local Linear Embedding [LLE].

## Dataset

There are 13 datasets available for us that dataset 1 to 13 had almost the same behavior. Therefore, I will present the results of analyzation of dataset 1 and 13 for their significant different performance. 

## Implementing

I will explain about the coding section and out procedure in coding. I had followed the below steps:
a) Dividing each dataset into train and test sections stablished in DatasourceReader class.
b) Dividing each dataset into 10 folds stablished in DatasourceReader class.
c) Creating functions that apply GridsearchCV on train and test sections to optimize the parameters of each classifier.
d) Creating function for each classifier KNN, RF, MLP, and RBF which are stablished in Utils class. For RBFNN I customized the original class to be able to use it in Python.
e) Creating function for each dimensionality reduction algorithms with each classifier separately for each fold, stablished in Calculation Class.
f) Creating function that includes each DR algorithm with each of the Classifiers will be called on each original dataset to calculate the explained variance (each DR algorithm and classifier is again optimized).


## Experimental

for splitting the data more efficiently, K-fold cross-validation has been used as a strong factor for validation. In the K-fold cross-validation all the entries in the original training dataset are managed for training as well as validation. In each dataset, there were some required tasks such as:

a) Analyzing the performance of each classifier for each dataset in the “Baseline” section, through two factors: F-measure and Accuracy. This has been done by considering 10 folds on each dataset.
b) calculating the variability explained by each extracted feature. For this purpose, I created a function that analyzes the dimensionality reduction algorithms by classifier to generate the optimal reduced-dimension matrix for calculating the ratio of variance of a feature to the variance of the features of the original data set without considering K-fold. The summation of these ratios should be then provided in the section “overall variability explained”.
c) Calculating the performance of each classifier on each dimensionality reduction algorithms in the “with dimensionality reduction” section through two factors of F-measure and Accuracy on each 10 folds of the dataset. In this regard, I designed a function that runs each classifier on each dimensionality reduction algorithms, we simultaneously calculated the ratio of the variance of each feature to the variance of the original dataset, and the optimized matrix of features. This process was so time-consuming that although different values of parameters has been tested, but I couldn’t achieve a good output.

## Conclusion

Dimensionality Reduction (DR) methods can be used in data pre-processing to achieve efficient data reduction. Using the dimensionality reduction methods, will probably improve the model performance. The reduced size of samples in dimensionality reduction leads to less calculation and much faster pace in performing the algorithm.
Although using DR algorithms on different datasets should have better results comparing to not using them, in our project, depending on the dimensions of the dataset, DR showed different accuracy. for most scale datasets I had a good performance.
