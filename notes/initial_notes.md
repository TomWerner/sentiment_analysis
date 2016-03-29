# Baseline
The baseline is a Bernoulli Naive Bayes model, using a boolean matrix of features/documents. Only single words were used
(no n-grams), and words that occurred only once were ignored. The result of 10-fold cross validation was an accuracy of **0.81, std: 0.01**  
The test accuracy was **0.82748**

### Baseline with counts
Identical to without counts for Bernoulli

# Data Selection
## Counts, TF-IDF, Sci-kit learn TF-IDF loader
We wanted to see if doing the TF-IDF transformation would be at all useful, or if we should stick with the boolean data.
As we can see, logistic regression (with L1 regularization on the TF-IDF transformed data performed far better than
other options, and so we proceeded using that model. Additionally, the TF-IDF data was better for all three
tested models.

Model | Input Type | 10-fold Accuracy | Stardard Dev
--- | --- | --- | ---
Bernoulli | Normal Inputs | 0.81 | 0.01
Multinomial | Normal Inputs | 0.79 | 0.02
LogReg | Normal Inputs | 0.85 | 0.01
Bernoulli | **TF-IDF Inputs** | **0.81** | 0.01
Multinomial | **TF-IDF Inputs** | **0.81** | 0.01
LogReg | **TF-IDF Inputs** | **0.87** | 0.01
Bernoulli | Sci-kit learn TF-IDF | 0.69 | 0.01
Multinomial | Sci-kit learn TF-IDF | 0.71 | 0.01
LogReg | Sci-kit learn TF-IDF | 0.71 | 0.01

# Variable Selection
## K-Best features using chi-squared test

![L1_LogReg](https://github.com/TomWerner/sentiment_analysis/blob/master/images/features_vs_accuracy_LogRegL1_0_to_max.png "L1 Regularized Logistic Regression Feature Count vs Accuracy")
![L1_LogReg](https://github.com/TomWerner/sentiment_analysis/blob/master/images/features_vs_accuracy_LogRegL1_0_to_4000.png "L1 Regularized Logistic Regression Feature Count vs Accuracy")
![L1_LogReg](https://github.com/TomWerner/sentiment_analysis/blob/master/images/features_vs_accuracy_LogRegL1_0_to_2500.png "L1 Regularized Logistic Regression Feature Count vs Accuracy")


Using these plots we found somewhere around 2000 features was ideal. The following 10-fold scores for models
are build using first the top 2000 chi-squared features, and then the L1 regularization as another layer of features
selection.

Model | 10-fold Accuracy | Standard Dev | 10-fold train/pred time (seconds)
--- | --- | --- | ---
BernoulliNB | 0.85 | 0.01 | 01.472
MultinomialNB | 0.87 | 0.01 | 00.956
LinearSVC | **0.89** | 0.00 | 02.895
LogisticRegression | 0.87 | 0.01 | 04.200
L1_LogisticRegression | 0.87 | 0.00 | 05.601
RandomForest | 0.78 | 0.01 | 47.391
RBF_SVC | 0.70 | 0.01 | 56 minutes
