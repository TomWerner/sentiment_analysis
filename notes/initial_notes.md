# Baseline
The baseline is a Bernoulli Naive Bayes model, using a boolean matrix of features/documents. Only single words were used
(no n-grams), and words that occurred only once were ignored. The result of 10-fold cross validation was
**Accuracy for Baseline BernoulliNB: 0.81, std: 0.01**
The test accuracy was **0.82748**

## Baseline with counts
Identical to without counts for Bernoulli

# Data Selection
## Counts, TF-IDF, Sci-kit learn TF-IDF loader
We wanted to see if doing the TF-IDF transformation would be at all useful, or if we should stick with the boolean data.
As we can see, logistic regression (with L1 regularization on the TF-IDF transformed data performed far better than
other options, and so we proceeded using that model. Additionally, the TF-IDF data was superior for all three
tested models.

Model | Input Type | 10-fold Accuracy | Stardard Dev
Bernoulli | Normal Inputs | 0.81 | 0.01
Multinomial | Normal Inputs | 0.79 | 0.02
LogReg | Normal Inputs | 0.85 | 0.01
Bernoulli | TF-IDF Inputs | 0.81 | 0.01
Multinomial | TF-IDF Inputs | 0.81 | 0.01
LogReg | TF-IDF Inputs | 0.87 | 0.01
Bernoulli | Sci-kit learn TF-IDF | 0.69 | 0.01
Multinomial | Sci-kit learn TF-IDF | 0.71 | 0.01
LogReg | Sci-kit learn TF-IDF | 0.71 | 0.01

# Variable Selection
## K-Best features using chi-squared test
See features_vs_accuracy plots for naive bayes and L1 logistic regression. Using these, the L1 regularized logistic
regression model proved better, and somewhere around 850 features was ideal. The following 10-fold scores for models
are build using first the top 850 chi-squared features, and then the L1 regularization as another layer of features
selection.

Model | 10-fold Accuracy | Standard Dev
BernoulliNB | 0.85 | 0.01
MultinomialNB | 0.87 | 0.01
LinearSVC | **0.88** | 0.00
LogisticRegression | 0.87 | 0.01
L1_LogisticRegression | 0.87 | 0.00
RandomForest | 0.78 | 0.01
RBF_SVC | 0.71 | 0.01
