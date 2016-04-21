# Logistic Regression for Sentiment Analysis
## What is Sentiment Analysis
I want to predict whether a given movie review is positive or negative. This could be used to track the sentiment
about a movie if it was trending on twitter for example, or to determine positivity of a movie's reviews on ImDB.
I'll be using logistic regression to predict if a review is positive (1) or negative (0).

## Data Representation
In order to work with the data, I needed to convert it from raw text into something more usable. The main steps I took were
phrase splitting, word cleaning and negation propagation, and vocabulary assembly.

 - **Phrase splitting**: Split each review on characters like ?.,!:; to identify "phrases"
 - **Word cleaning**: Remove all the junk from words - every single character here was used at least once. "$%^&*()#@\'+-/0123456789<>=\\[]_~{}|`
 - **Negation propagation**: I used an idea from [2] to handle negations. If I saw "not", "n't", or "no" in a phrase, I negated it. So
 an example sentence "The movie, while not good, was decent", would be extracted as ["the", "movie", "not_while", "not_not", "not_good", "was", "decent"].
 You can see how the middle phrase was negated. This allows us to handle things like that, where "not good" are used.
 - **Vocabulary assembly**: I needed a global list of words in the training dataset to build the input matrix,
 where input[i][j] is the count of the j'th word in the i'th review.

## Data Representation Options
When representing the data, I had two options. I could represent it all as binary values, so if word j was present in a review,
Xj=1. This is less useful than counts, so Xj={count of word j}. I'll call that normal inputs. The other options is to
use the TF-IDF transformation on this data matrix. It uses the frequency of a word in a document (term frequency), emphasizing it
more, like with the counts. However, it also uses the inverse of the document frequency (IDF), so that words that are common
across all the documents (reviews) are emphasized less, like "the". In order to decide which to use, I'll look at the cross
validation score of a logistic regression model on the data.

Model | Input type | Accuracy | Std
--- | --- | --- | ---
Logistic Regression L2 | Normal Inputs | 0.85 | 0.01
Logistic Regression L1 | Normal Inputs | 0.85 | 0.01
Logistic Regression L2 | TF-IDF Inputs | 0.87 | 0.01
Logistic Regression L1 | TF-IDF Inputs | 0.87 | 0.01

```python
# Load the data
inputs, outputs, word_list = pickle.load(open(training_data_filename, 'rb'))
# Get a copy of the inputs using the TF-IDF transformation
tfidf_inputs = TfidfTransformer().fit_transform(inputs)

# Check all the combinations
for input_matrix, name in zip([inputs, tfidf_inputs], ['Normal Inputs', 'TF-IDF Inputs']):
    for model, model_name in zip([LogisticRegression(), LogisticRegression(penalty='l1')],
                                 ['LogReg', 'LogReg_L1']):
        # Do 10-fold cross validation with the selected model and inputs, then output the results
        scores = cross_val_score(model, input_matrix, outputs.ravel(), cv=10)
        logging.info("%s | %s | %.02f | %.02f" % (model_name, name, scores.mean(), scores.std()))
```

Based on these results, I decided to move forwards using the TF-IDF inputs.

## Feature Selection
The next step is to do feature selection (or model selection). I needed to determine which variables are useful for the model.
Not only will this help increase the accuracy of the model, and make it train faster, its also important to make sure
that our predictions can generalize. For example, imagine that we trained on these two reviews.

 - Terminator was awesome! (positive)
 - I hated that movie. (negative)

If we used all of these words, including "terminator", and "movie", its possible that we would misclassify

 - **I** loved **that movie**.

Notice how "I", "that", and "movie" appear in the negative review, but don't have much to do with negativity. Likewise, if we used
all 60,000 or so words in our model, we'd likely hit strange cases like that.

To do the feature selection, I used a a chi-squared test to rank the features, and then selected 1000, 5000, etc, up to 60,000 features, generating this plot.
The accuracy comes from building a logistic regression model using those top k features.

![L1_LogReg](https://github.com/TomWerner/sentiment_analysis/blob/lin_reg/images/lin_reg_features_vs_accuracy_LogReg_0_to_max.png "Chi-squared feature selection for Logistic Regression Feature Count vs Accuracy")

Based on this, I selected around 40,000 features.


## Model Selection
Now that I had our feature count, I needed to pick a model to use. I was choosing between Logistic Regression with
l2 regularization and Logistic Regression with L1 regularization, which results in a sparse solution.

Model | Accuracy | std
--- | --- | ---
Logistic Regression L2 | 0.87 | 0.01
Logistic Regression L1 | 0.87 | 0.01

```python
# I'll use variations of this for all model comparisons
def evaluate_var_selected_models(inputs, outputs, num_vars, l1_step=LogisticRegression('l1')):
    models = [LogisticRegression(penalty='l2'), LogisticRegression(penalty='l1')]
    model_names = ["LogReg", 'LogReg_L1']
    for model, model_name in zip(models, model_names):
        pipeline = Pipeline([('chi2_top_k', SelectKBest(chi2, num_vars)), (model_name, model) ])
        scores = cross_val_score(pipeline, inputs, outputs.ravel(), cv=10, n_jobs=5)
        logging.info("%s | %.02f | %.02f" % (model_name, scores.mean(), scores.std()))
```

We can see that the accuracy of both models are the same, lets try changing the complexity (how strict is the regularization).


Model | Accuracy | std
--- | --- | ---
Logistic Regression L1, C=.01 | 0.50 | 0.00
Logistic Regression L1, C=.1 | 0.79 | 0.01
Logistic Regression L1, C=1 | 0.87 | 0.01
Logistic Regression L1, C=1 | 0.87 | 0.01
Logistic Regression L1, C=2 | 0.87 | 0.01
Logistic Regression L1, C=3 | 0.87 | 0.01
Logistic Regression L1, C=4 | 0.87 | 0.01
Logistic Regression L1, C=5 | 0.86 | 0.01
Logistic Regression L1, C=10 | 0.85 | 0.01

Model | Accuracy | std
--- | --- | ---
Logistic Regression L2, C=.01 | 0.78 | 0.01
Logistic Regression L2, C=.1 | 0.84 | 0.01
Logistic Regression L2, C=1 | 0.87 | 0.01
Logistic Regression L2, C=1 | 0.87 | 0.01
Logistic Regression L2, C=2 | 0.87 | 0.01
Logistic Regression L2, C=3 | 0.87 | 0.01
Logistic Regression L2, C=4 | 0.87 | 0.01
Logistic Regression L2, C=5 | 0.87 | 0.01
Logistic Regression L2, C=10 | 0.86 | 0.01

Changing C doesn't seem to have much of an effect, and we don't get any better than 87% accuracy. If I had to stop here,
I'd choose the L1 regularized model, since I like the sparsity, but I want to try N-grams.

## N-Grams
Another technique to improve the accuracy of sentiment analysis is the inclusion of N-Grams, which capture more than just
individual words. This is useful in a sentence like "Thor was very good". The two grams are "Thor was", "was very", "very good".
"Very good" gives us more information than either word on their own. Using 2 and 3 grams does dramatically increase the
number of features, so I re-evaluated our data input choice.

## Data Representation
Model | Input type | Accuracy | Std
--- | --- | --- | ---
Logistic Regression L2 | Normal Inputs | 0.88 | 0.01
Logistic Regression L1 | Normal Inputs | 0.87 | 0.01
Logistic Regression L2 | TF-IDF Inputs | 0.88 | 0.01
Logistic Regression L1 | TF-IDF Inputs | 0.86 | 0.01

TF-IDF still looks good, and it theoretically should perform better, so I'll stay with it.

## Feature selection
Feature selection is even more important here, since there are around 1,000,000 possible features now. I'll use the same
approach.

![L1_LogReg](https://github.com/TomWerner/sentiment_analysis/blob/lin_reg/images/lin_reg_features_vs_accuracy_LogReg_0_to_max.png "Chi-squared feature selection for Logistic Regression Feature Count vs Accuracy")


From this graph, somewhere around 600,000 features looks decent.

# Model selection
I also repeated model selection using these 600,00 features.

Model | Input type | Accuracy | Std
--- | --- | --- | ---
Logistic Regression L1, C=0.01 | 0.50 | 0.00
Logistic Regression L1, C=0.1 | 0.73 | 0.01
Logistic Regression L1, C=1 | 0.86 | 0.01
Logistic Regression L1, C=10 | 0.87 | 0.01
Logistic Regression L2, C=0.01 | 0.78 | 0.01
Logistic Regression L2, C=0.1 | 0.83 | 0.01
Logistic Regression L2, C=1 | 0.88 | 0.01
Logistic Regression L2, C=10 | 0.89 | 0.01

We've managed to beat our previous best, and achieve 89% accuracy!

## Test performance
Now that I've selected my final model and complexity setting, lets see how this model does at predicting the
sentiment of the test set.

Test accuracy: **0.88428**


