# Baseline
The baseline is a Bernoulli Naive Bayes model, using a boolean matrix of features/documents. Only single words were used
(no n-grams), and words that occurred only once were ignored. The result of 10-fold cross validation was an accuracy of **0.81, std: 0.01**  
The test accuracy was **0.82748**

### Baseline with counts
Identical to without counts for Bernoulli

## Data Selection
### Counts, TF-IDF, Sci-kit learn TF-IDF loader
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

## Variable Selection
### K-Best features using chi-squared test

![L1_LogReg](https://github.com/TomWerner/sentiment_analysis/blob/master/images/features_vs_accuracy_LogRegL1_0_to_max.png "L1 Regularized Logistic Regression Feature Count vs Accuracy")
![L1_LogReg](https://github.com/TomWerner/sentiment_analysis/blob/master/images/features_vs_accuracy_LogRegL1_0_to_4000.png "L1 Regularized Logistic Regression Feature Count vs Accuracy")
![L1_LogReg](https://github.com/TomWerner/sentiment_analysis/blob/master/images/features_vs_accuracy_LogRegL1_0_to_2500.png "L1 Regularized Logistic Regression Feature Count vs Accuracy")


Using these plots we found somewhere around 2000 features was ideal. The following 10-fold scores for models
are build using first the top 2000 chi-squared features, and then the L1 regularized logistic regression
as another layer of features selection.

Model | 10-fold Accuracy | Standard Dev | 10-fold train/pred time (seconds)
--- | --- | --- | ---
BernoulliNB | 0.84 | 0.01 | 01.472
MultinomialNB | 0.85 | 0.01 | 00.956
LinearSVC | 0.87 | 0.01 | 02.895
L1_LinearSVC | 0.87 | 0.01
LogisticRegression | 0.86 | 0.01 | 04.200
L1_LogisticRegression | 0.87 | 0.01 | 05.601
RandomForest | 0.77 | 0.01 | 47.391
RBF_SVC | 0.70 | 0.01 | 56 minutes

As a sidenote, the extreme time it took to train the RBF SVC, and its underperformance means it will be excluded from
future analysis.

# N-Grams
2 and 3-grams add more variables to the model by including pairs of words. For example, with the sentence:
"This movie was awesome", the 2-grams are "This movie", "movie was", and "was awesome". This could potentially add
information in cases like "very good", where "very" and "good" separately don't necessarily add as much information.

## Data selection
Because we added so many more features by including 2 and 3-grams, we wanted to make sure that using the TF-IDF was
still more useful than just the counts. We left out the sklearn tfidf because of how poorly it performed earlier.

Model | Input Type | 10-fold Accuracy | Stardard Dev
--- | --- | --- | ---
Bernoulli | Normal Inputs | 0.86 | 0.01
Multinomial | Normal Inputs | 0.85 | 0.01
LogReg | Normal Inputs | 0.87 | 0.01
L1_LinearSVC | Normal Inputs | 0.86 | 0.01
LinearSVC | Normal Inputs | 0.88 | 0.01
Bernoulli | TF-IDF Inputs | 0.86 | 0.01
Multinomial | TF-IDF Inputs | 0.86 | 0.01
LogReg | TF-IDF Inputs | 0.86 | 0.01
L1_LinearSVC | TF-IDF Inputs | 0.88 | 0.01
LinearSVC | TF-IDF Inputs | **0.89** | 0.01

We can see that using the TF-IDF transformation on the 3-gram data yields the best results. Now we want to repeat
the process of variable selection to see if we reach comparable results on a subset of the variables. When we use
all the variables, even with cross validation our accuracy will be slightly inflated because all folds will have a
common "vocabulary", while our test set does not. So besides the speed increases and interpretability increases that
variable selection brings, it also helps us generalize better.

## K-Best features using chi-squared test
Because adding 3-Grams adds so many more variables, we want to repeat the process of checking accuracy vs feature count,
using the top k features from a chi-squared test.

![L1_LinSVC_3_gram](https://github.com/TomWerner/sentiment_analysis/blob/master/images/features_vs_accuracy_3_gram_LinSVCL1_0_to_max.png "L1 Regularized Linear SVC Feature Count vs Accuracy")
![L1_LinSVC_3_gram](https://github.com/TomWerner/sentiment_analysis/blob/master/images/features_vs_accuracy_3_gram_LinSVCL1_0_to_100k.png "L1 Regularized Linear SVC Feature Count vs Accuracy")
![L1_LinSVC_3_gram](https://github.com/TomWerner/sentiment_analysis/blob/master/images/features_vs_accuracy_3_gram_LinSVCL1_0_to_40k.png "L1 Regularized Linear SVC Feature Count vs Accuracy")

Here we can see that somewhere around 13000 features looks reasonable.
The following table then shows our results of trying various models after applying the chi-squared ranking feature
selection step followed by a L1 model based feature selection step. Rows that achieve at least 88% are bolded.

L1 Feature Selection Step | Model | 10-fold Accuracy | Standard Dev
--- | --- | --- | ---
LogReg L1 | BernoulliNB | 0.84 | 0.01
LogReg L1 | MultinomialNB | 0.86 | 0.01
LogReg L1 | LinearSVC | 0.87 | 0.01
LogReg L1 | L1_LinearSVC | 0.87 | 0.01
LogReg L1 | LogisticRegression | 0.84 | 0.01
LogReg L1 | L1_LogisticRegression | 0.86 | 0.01
Linear SVC L1, C=.1 | BernoulliNB | 0.82 | 0.01
Linear SVC L1, C=.1 | MultinomialNB | 0.84 | 0.01
Linear SVC L1, C=.1 | LinearSVC | 0.84 | 0.01
Linear SVC L1, C=.1 | L1_LinearSVC | 0.85 | 0.01
Linear SVC L1, C=.1 | LogisticRegression | 0.82 | 0.01
Linear SVC L1, C=.1 | L1_LogisticRegression | 0.85 | 0.01
Linear SVC L1, C=1 | BernoulliNB | 0.86 | 0.01
Linear SVC L1, C=1 | MultinomialNB | 0.86 | 0.01
**Linear SVC L1, C=1** | **LinearSVC** | **0.88** | **0.01**
**Linear SVC L1, C=1** | **L1_LinearSVC** | **0.88** | **0.01**
Linear SVC L1, C=1 | LogisticRegression | 0.86 | 0.01
Linear SVC L1, C=1 | L1_LogisticRegression | 0.86 | 0.01
Linear SVC L1, C=10 | BernoulliNB | 0.86 | 0.01
Linear SVC L1, C=10 | MultinomialNB | 0.86 | 0.01
**Linear SVC L1, C=10** | **LinearSVC** | **0.88** | **0.01**
**Linear SVC L1, C=10** | **L1_LinearSVC** | **0.88** | **0.01**
Linear SVC L1, C=10 | LogisticRegression | 0.86 | 0.01
Linear SVC L1, C=10 | L1_LogisticRegression | 0.86 | 0.01

We can see that the LinearSVC and L1_LinearSVC perform the best, and they seem to be fairly agnostic to variations of
the C parameter in the L1 feature selection stage. We can also see that the L1_LinearSVC selector does better than
the logistic regression one.

We'll now examine the effect of the C parameter on the LinearSVC and L1_linearSVC as models after the feature selection.

L1 Feature Selection Step | Model | 10-fold Accuracy | Standard Dev
--- | --- | --- | ---
Linear SVC L1, C=1 | LinearSVC C=.001 | 0.76 | 0.01
**Linear SVC L1, C=1** | **LinearSVC C=1** | **0.88** | **0.01**
Linear SVC L1, C=1 | LinearSVC C=.1 | 0.86 | 0.01
**Linear SVC L1, C=1** | **LinearSVC C=10** | **0.88** | **0.01**
Linear SVC L1, C=1 | LinearSVC C=100 | 0.87 | 0.01
Linear SVC L1, C=1 | L1_LinearSVC C=.001 | 0.50 | 0.00
**Linear SVC L1, C=1** | **L1_LinearSVC C=1** | **0.88** | **0.01**
Linear SVC L1, C=1 | L1_LinearSVC C=.1 | 0.83 | 0.01
Linear SVC L1, C=1 | L1_LinearSVC C=10 | 0.87 | 0.01
Linear SVC L1, C=1 | L1_LinearSVC C=100 | 0.86 | 0.01

Again, the models seem fairly agnostic to it - without new methods we seem to have maxed out our model. To recap our
best method and our evaluation technique:
Repeat the following for each fold

**1.** chi-squared test on sparse tdidf dataset to select top 13000 features from the training data

**2.** Use a Linear SVC with L1 regularization, C parameter set to 1 to perform further feature selection.

**3.** Train a Linear SVC on the resulting feature set, with L2 regularization, C parameter set to 1.

**4.** Use this model to make a prediction on the test fold, our average performance was **88%** on an average of **1697** variables. (std = 18)

**5.** Use this model to make a prediction on the entire test set. **88.052%** accuracy on 25,000 reviews!

Here's a few of the features that were deemed useful: 'a', 'a_better', 'a_bit', 'a_boy', 'a_bunch_of', 'a_car', 'a_complete', 'a_copy', 'a_dvd', 'a_favor', 'a_great_performance', 'a_group_of', 'a_job', 'a_joke', 'a_little', 'a_look', 'a_masterpiece'

We also used the ExtraTreesClassifier's ability to rank feature to get a rough idea of some of the top words:
'bad', 'great', 'awful', 'worst', 'the_worst', 'not_even', 'of_the_best', 'and', 'just', 'nothing', 'boring', 'minutes', 'no', 'waste', 'not_this', 'excellent', 'love', 'poor'
are all very reasonable choices!


ELM: full linear | 0.84 | 0.01
ELM: full linear, 1 tanh | 0.86 | 0.01
ELM: full linear, 5 tanh | 0.87 | 0.01
ELM: full linear, 10 tanh | 0.87 | 0.01
ELM: full linear, 100 tanh | 0.86 | 0.01
ELM: full linear, 500 tanh | 0.86 | 0.01
ELM: full linear, 1000 tanh | 0.86 | 0.01
ELM: full linear, 1500 tanh | 0.86 | 0.01

ELM: 100 linear | 0.64 | 0.01
ELM: 100 linear, 10 tanh | 0.66 | 0.01
ELM: 100 linear, 100 tanh | 0.74 | 0.01

ELM: 500 linear | 0.75 | 0.01
ELM: 500 linear, 10 tanh | 0.80 | 0.01
ELM: 500 linear, 100 tanh | 0.82 | 0.01
