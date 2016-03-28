# Group 3 Sentiment Analysis
Large Data Analysis Project

## Get the data
The Large Movie Review Dataset can be downloaded here: http://ai.stanford.edu/~amaas/data/sentiment/  
Extract the data so that aclImdb/ is in the root directory of the project (sentiment_analysis/aclImdb)  

### A good starting paper
I used this paper (http://arxiv.org/pdf/1305.6143.pdf) to get a start on a lot of the code/modeling, including the negation.

## Build the training/testing datasets
Run from the main directory ("python build_saved_data_files.py" will do this)
```python
trn_inputs, trn_outputs, trn_word_list = 
  preprocessing.build_data_target_matrices("aclImdb/train/pos/", "aclImdb/train/neg/", save_data=True)
tst_inputs, tst_outputs, _ = 
  preprocessing.build_test_data_target_matrices("aclImdb/test/pos/", "aclImdb/test/neg/", trn_word_list)
```
This will create training_data.pkl and testing_data.pkl  
Then you can train and evaluate a model using the following code (also in naive_bayes.py):  
```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix

inputs, outputs, _ = pickle.load(open("training_data.pkl", 'rb'))
test_inputs, test_outputs, _ = pickle.load(open("testing_data.pkl", 'rb'))

model = BernoulliNB()
model.fit(inputs, outputs.ravel())
print(confusion_matrix(test_outputs, model.predict(test_inputs)))
```

`model.fit(inputs, outputs.ravel())` should work with any of the sklearn models that support sparse matrices.  
We should also be able to use any of the sklearn feature selection techniques on our data, which we'll
definitely want to do. For example, "the", "and", "or" will probably be extremely common, but don't give much information.
