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
 - **Negation propagation**: We used an idea from [2] to handle negations. If we saw "not", "n't", or "no" in a phrase, we negated it. So
 an example sentence "The movie, while not good, was decent", would be extracted as ["the", "movie", "not_while", "not_not", "not_good", "was", "decent"].
 You can see how the middle phrase was negated. This allows us to handle things like that, where "not good" are used.
 - **Vocabulary assembly**: We needed a global list of words in our training dataset to build our input matrix,
 where input[i][j] is the count of the j'th word in the i'th review.

## Data Representation Options
When representing the data, I had two options. I could represent it all as binary values, so if word j was present in a review,
Xj=1. This is less useful than counts, so Xj=<count of word j>. I'll call that normal inputs. The other options is to
use the TF-IDF transformation on this data matrix. It uses the frequency of a word in a document (term frequency), emphasizing it
more, like with the counts. However, it also uses the inverse of the document frequency (IDF), so that words that are common
across all the documents (reviews) are emphasized less, like "the". In order to decide which to use, I'll look at the cross
validation score of a logistic regression model on the data.

Model | Input type | Accuracy | Std
--- | --- | --- | ---
LogReg | Normal Inputs | 0.85 | 0.01
LogReg_L1 | Normal Inputs | 0.85 | 0.01
LogReg | TF-IDF Inputs | 0.87 | 0.01
LogReg_L1 | TF-IDF Inputs | 0.87 | 0.01

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
The next step is to do feature selection (or model selection). I needed to determine which variables

Feature selection
```python
def chi2_cv_feature_selection(inputs, outputs, model, max_features, filename, k=10):
    x_values = np.linspace(1, max_features, num=40, dtype=int)
    kf = KFold(inputs.shape[0], n_folds=k, shuffle=True)
    validation_y_values = []
    training_y_values = []
    for train_indices, val_indices in kf:
        chi_2_cv_feature_selection_helper(inputs, outputs, model, train_indices, val_indices, training_y_values,
                                          validation_y_values, x_values)

    plt.clf()
    sns.tsplot(data=validation_y_values, time=x_values, ci=[68, 95], color='red', condition="Validation Accuracy")
    sns.tsplot(data=validation_y_values, time=x_values, ci=[68, 95], color='blue', condition="Training Accuracy")

    plt.title("Feature Count vs Feature Count (selected with chi2)")
    plt.xlabel("Feature Count (top k using chi2)")
    plt.ylabel("Accuracy")
    plt.ylim((.8, .9))
    plt.savefig(filename)

chi2_cv_feature_selection(inputs, outputs, LogisticRegression(), inputs.shape[1], "images/lin_reg_features_vs_accuracy_LogReg_0_to_max.png")
```

40,000 looks decent

```python
def evaluate_var_selected_models(inputs, outputs, num_vars, l1_step=LogisticRegression('l1')):
    models = [LogisticRegression(penalty='l2'), LogisticRegression(penalty='l1')]
    model_names = ["LogReg", 'LogReg_L1']
    for model, model_name in zip(models, model_names):
        pipeline = Pipeline([('chi2_top_k', SelectKBest(chi2, num_vars)),
                             (model_name, model)
                             ])

        scores = cross_val_score(pipeline, inputs, outputs.ravel(), cv=10, n_jobs=5)
        logging.info("%s | %.02f | %.02f" % (model_name, scores.mean(), scores.std()))

2016-04-20 18:59:29,191 [MainThread  ] [INFO ]  LogReg | 0.87 | 0.01
2016-04-20 18:59:36,432 [MainThread  ] [INFO ]  LogReg_L1 | 0.87 | 0.01
```

I like L1 better, sparse is good, lets tweak it

```python
def evaluate_var_selected_models(inputs, outputs, num_vars, l1_step=LogisticRegression('l1')):
    models = [LogisticRegression(penalty='l1', C=.01), LogisticRegression(penalty='l1', C=.1), LogisticRegression(penalty='l1', C=1), LogisticRegression(penalty='l1', C=10)]
    model_names = ['LogReg_L1, C=.01', 'LogReg_L1, C=.1', 'LogReg_L1, C=1', 'LogReg_L1, C=10']
    for model, model_name in zip(models, model_names):
        pipeline = Pipeline([('chi2_top_k', SelectKBest(chi2, num_vars)),
                             # ('l1_step', SelectFromModel(l1_step)),
                             (model_name, model)
                             ])

        scores = cross_val_score(pipeline, inputs, outputs.ravel(), cv=10, n_jobs=5)
        logging.info("%s | %.02f | %.02f" % (model_name, scores.mean(), scores.std()))
def evaluate_var_selected_models(inputs, outputs, num_vars, l1_step=LogisticRegression('l1')):
    models = [LogisticRegression(penalty='l2', C=.01), LogisticRegression(penalty='l2', C=.1), LogisticRegression(penalty='l2', C=1), LogisticRegression(penalty='l2', C=10)]
    model_names = ['LogReg, C=.01', 'LogReg, C=.1', 'LogReg, C=1', 'LogReg, C=10']
    for model, model_name in zip(models, model_names):
        pipeline = Pipeline([('chi2_top_k', SelectKBest(chi2, num_vars)),
                             # ('l1_step', SelectFromModel(l1_step)),
                             (model_name, model)
                             ])

        scores = cross_val_score(pipeline, inputs, outputs.ravel(), cv=10, n_jobs=5)
        logging.info("%s | %.02f | %.02f" % (model_name, scores.mean(), scores.std()))


2016-04-20 19:03:28,916 [MainThread  ] [INFO ]  LogReg_L1, C=.01 | 0.50 | 0.00
2016-04-20 19:03:34,377 [MainThread  ] [INFO ]  LogReg_L1, C=.1 | 0.79 | 0.01
2016-04-20 19:03:41,572 [MainThread  ] [INFO ]  LogReg_L1, C=1 | 0.87 | 0.01
2016-04-20 19:03:53,322 [MainThread  ] [INFO ]  LogReg_L1, C=10 | 0.85 | 0.01

2016-04-20 19:03:41,572 [MainThread  ] [INFO ]  LogReg_L1, C=1 | 0.87 | 0.01
2016-04-20 19:08:31,214 [MainThread  ] [INFO ]  LogReg_L1, C=2 | 0.87 | 0.01
2016-04-20 19:08:38,891 [MainThread  ] [INFO ]  LogReg_L1, C=3 | 0.87 | 0.01
2016-04-20 19:08:47,676 [MainThread  ] [INFO ]  LogReg_L1, C=4 | 0.87 | 0.01
2016-04-20 19:08:57,285 [MainThread  ] [INFO ]  LogReg_L1, C=5 | 0.86 | 0.01


2016-04-20 19:05:35,668 [MainThread  ] [INFO ]  LogReg, C=.01 | 0.78 | 0.01
2016-04-20 19:05:41,882 [MainThread  ] [INFO ]  LogReg, C=.1 | 0.84 | 0.01
2016-04-20 19:05:50,078 [MainThread  ] [INFO ]  LogReg, C=1 | 0.87 | 0.01
2016-04-20 19:06:03,186 [MainThread  ] [INFO ]  LogReg, C=10 | 0.86 | 0.01

2016-04-20 19:05:50,078 [MainThread  ] [INFO ]  LogReg, C=1 | 0.87 | 0.01
2016-04-20 19:06:53,817 [MainThread  ] [INFO ]  LogReg, C=2 | 0.87 | 0.01
2016-04-20 19:07:04,488 [MainThread  ] [INFO ]  LogReg, C=3 | 0.87 | 0.01
2016-04-20 19:07:14,592 [MainThread  ] [INFO ]  LogReg, C=4 | 0.87 | 0.01
2016-04-20 19:07:25,162 [MainThread  ] [INFO ]  LogReg, C=5 | 0.87 | 0.01

# we've maxed out here, I'd choose the l1 model since it'll be sparser
```


## Data Representation
1,2,3 grams:
2016-04-20 15:43:29,016 [MainThread  ] [INFO ]  Data loaded
2016-04-20 15:50:52,394 [MainThread  ] [INFO ]  LogReg | Normal Inputs | 0.88 | 0.01
2016-04-20 15:52:15,280 [MainThread  ] [INFO ]  LogReg_L1 | Normal Inputs | 0.87 | 0.01
2016-04-20 15:53:11,171 [MainThread  ] [INFO ]  LogReg | TF-IDF Inputs | 0.88 | 0.01
2016-04-20 15:53:54,571 [MainThread  ] [INFO ]  LogReg_L1 | TF-IDF Inputs | 0.86 | 0.01

Go forwards with TF-IDF

Feature selection
```python
chi2_cv_feature_selection(inputs, outputs, LogisticRegression(), inputs.shape[1], "images/lin_reg_features_vs_accuracy_LogReg_0_to_max_3_gram.png")
```

600,000 looks decent


2016-04-20 20:48:40,103 [MainThread  ] [INFO ]  LogReg_l1C=0.01 | 0.50 | 0.00
2016-04-20 20:49:06,341 [MainThread  ] [INFO ]  LogReg_l1C=0.1 | 0.73 | 0.01
2016-04-20 20:49:31,747 [MainThread  ] [INFO ]  LogReg_l1C=1 | 0.86 | 0.01
2016-04-20 20:50:03,481 [MainThread  ] [INFO ]  LogReg_l1C=10 | 0.87 | 0.01
2016-04-20 20:50:29,133 [MainThread  ] [INFO ]  LogReg_l2C=0.01 | 0.78 | 0.01
2016-04-20 20:51:00,677 [MainThread  ] [INFO ]  LogReg_l2C=0.1 | 0.83 | 0.01
2016-04-20 20:51:42,873 [MainThread  ] [INFO ]  LogReg_l2C=1 | 0.88 | 0.01
2016-04-20 20:52:47,810 [MainThread  ] [INFO ]  LogReg_l2C=10 | 0.89 | 0.01

Clearly LogReg with l2 regularization and C=10 is best


Now lets check out the accuracy on the test set
Test accuracy: 0.88428





