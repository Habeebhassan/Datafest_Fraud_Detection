# Online Fraud Detection

## Introduction

Online transactions have become increasingly popular, and this trend is expected to continue in the future, according to various surveys and research. However, this growth has also led to an increase in fraudulent transactions. Despite the implementation of various security systems, a significant amount of money is still being lost due to fraudulent transactions. Online fraud transactions occur when a person uses someone else’s credit card for personal reasons without the owner or the card-issuing authorities being aware of it. This project aims to address this issue.

## Project Scope

The Online Fraud Transaction Detection System is an extension of an existing system. The algorithms built using this system will go through the dataset and provide the appropriate output. In the long run, this system will be beneficial as it provides an efficient way to create a secure transaction system to analyze and detect fraudulent transactions. The Proposed algorithm algorithm used in this project is XGBOOST. Xgboost algorithm is a popular and efficient open-source implementation of the gradient boosted trees algorithm. Gradient boosting is a supervised learning algorithm that attempts to accurately predict a target variable by combining the estimates of a set of simpler, weaker models. This accuracy can be increased further by providing a huge dataset for model training. The scope of this application is far-reaching, and it can be used to detect the features of fraud transactions in datasets that are applicable in various sectors such as banking, insurance, e-commerce, money transfer, bill payments, etc. This will help increase security.

Work Flow

Load DataSet
Data Preprocessing
Feature Selection/Feature Engineering
Classification
XGboost Model Training
Prediction
Evaluation
After preprocessing and feature engineering is done, I'll divide the data into two sets: a training set and a separate test set that will be used exclusively at the final stage of model development. This ensures an unbiased assessment of the model's performance on entirely new data.

Next, I'll go through the following steps to construct and assess the model:

Hyperparameter Tuning with Cross Validation: Given the considerable number of hyperparameters in XGBoost, I'll employ Bayesian hyperparameter tuning. This approach is more efficient compared to grid or random search. I'll use (stratified) K-fold cross validation to identify the combination of hyperparameters that yields the highest cross-validated Conditional Recall score.

Determining Thresholds: The refined classifier from step one can assign a probability score to any given example. To classify an example, a probability threshold must be chosen. This threshold separates positive (fraud) and negative examples. While the standard practice is to set the threshold at 0.5, I'll explore empirical thresholding to strike a balance between precision and recall. This may lead to a higher recall rate by selecting an appropriate classification threshold.

Training and Testing: Using the entire training set, I'll train the model and then assess its performance on the test set using the discussed Conditional Recall metric.

Additionally, I'll conclude by comparing the performance of this model with a few other algorithms. While the ideal approach would involve nested cross-validation for comparing different models (e.g., XGBoost vs. Logistic Regression), this is computationally demanding. Therefore, I'll report the performance of these models based on a single test set.

## Proposed Algorithm

The XGBoost algorithm, short for Extreme Gradient Boosting, is a powerful implementation of the gradient boosting method. Gradient boosting is an ensemble learning technique that sequentially trains simple models, often shallow decision trees, with a focus on areas of the data that haven't been well predicted so far. The final prediction of the model is a weighted combination of these weak learners. XGBoost has demonstrated remarkable effectiveness in various regression and classification tasks.

For this task, I'll employ the XGBClassifier class from the xgboost package's Python API. This classifier utilizes the Extreme Gradient Boosting algorithm to optimize a specified loss function. In this case, I'll be using a logistic loss function, which is also the default choice in xgboost for two-class classification tasks.

## Bayesian Hyper-parameter Optimization

Bayesian hyper-parameter optimization commences by assuming a prior distribution for the model's parameters. In each iteration, it seeks to glean insights from previously assessed parameter values, updating this distribution to select values more likely to yield high scores in future trials. This stands in contrast to random and grid search methods, which explore the parameter space without leveraging past trials for guidance. As a result, Bayesian hyper-parameter tuning has demonstrated superior efficiency compared to both random search and grid search, particularly in scenarios where evaluating the objective function is resource-intensive and time-consuming, or when dealing with a high-dimensional parameter-space.

In the visual representation, the grey dots represent results from random search, while the green dots depict outcomes from Bayesian optimization (utilizing the Tree Parzen Estimator or TPE). Each dot signifies the lowest validation set error achieved within a specific number of trials for the respective method. It's evident that Bayesian optimization outperforms random search, and it does so with fewer iterations.

## Optimizing Hyperparameters Using Hyperopt

For fine-tuning the model's hyperparameters, I'll employ the Hyperopt package, which employs Bayesian optimization technique
