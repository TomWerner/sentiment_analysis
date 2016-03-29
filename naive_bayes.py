from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import pickle
import logging

import preprocessing
import utilities
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
import os
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier


def add_to_feature_map(map, key, data):
    list_obj = []
    if key in map.keys():
        list_obj = map[key]
    list_obj.append(data)
    map[key] = list_obj


def do_loo_l1_variable_selection(inputs, outputs, word_list):
    feature_list_map = {}
    log_reg = LogisticRegression(penalty='l1')
    kf = KFold(inputs.shape[0], n_folds=10, shuffle=True)
    for train, validation in kf:
        train_inputs = inputs[train]
        train_outputs = outputs[train]
        val_inputs = inputs[validation]
        val_outputs = outputs[validation]

        log_reg.fit(train_inputs, train_outputs.ravel())

        used_variables = np.where(log_reg.coef_[0] != 0)[0]
        add_to_feature_map(feature_list_map, 'log_reg', set(used_variables))
    for key in feature_list_map.keys():
        intersection = set.intersection(*feature_list_map[key])
        print("LOO Features for %s: %s" % (key, len(intersection)))
        print(", ".join([word_list[i] for i in intersection]))


def main():
    inputs, outputs, word_list = pickle.load(open("training_data.pkl", 'rb'))
    # test_inputs, test_outputs, _ = pickle.load(open("testing_data.pkl", 'rb'))
    logging.info("Data loaded")

    inputs = TfidfTransformer().fit_transform(inputs)


    # do_loo_l1_variable_selection()

    # chi2_cv_feature_selection(tfidf_inputs, outputs, LogisticRegression(penalty='l1'), inputs.shape[1], "images/features_vs_accuracy_NB_0_to_max.png")
    # chi2_cv_feature_selection(inputs, outputs, model, 10000, "images/features_vs_accuracy_NB_0_to_10000.png")
    # chi2_cv_feature_selection(inputs, outputs, model, 6000, "images/features_vs_accuracy_NB_0_to_6000.png")

    # chi2_cv_feature_selection(tfidf_inputs, outputs, LogisticRegression(penalty='l1'), inputs.shape[1], "images/features_vs_accuracy_LogRegL1_0_to_max.png")
    # chi2_cv_feature_selection(tfidf_inputs, outputs, LogisticRegression(penalty='l1'), 4000, "images/features_vs_accuracy_LogRegL1_0_to_4000.png")
    # chi2_cv_feature_selection(tfidf_inputs, outputs, LogisticRegression(penalty='l1'), 1000, "images/features_vs_accuracy_LogRegL1_0_to_1000.png")
    # chi2_cv_feature_selection(inputs, outputs, LogisticRegression(penalty='l1'), 2500, "images/features_vs_accuracy_LogRegL1_0_to_2500.png")

    evaluate_var_selected_models(inputs, outputs)


def evaluate_var_selected_models(inputs, outputs):
    models = [BernoulliNB(), MultinomialNB(), LinearSVC(), LogisticRegression(), LogisticRegression('l1'),
              RandomForestClassifier(), SVC()]
    model_names = ["BernoulliNB", "MultinomialNB", "LinearSVC", "LogisticRegression", "L1_LogisticRegression",
                   "RandomForest", "RBF_SVC"]
    for model, model_name in zip(models, model_names):
        pipeline = Pipeline([('chi2_top_2000', SelectKBest(chi2, 2000)),
                             ('l1_log_reg', SelectFromModel(LogisticRegression('l1')))])
        train_inputs = pipeline.fit_transform(inputs, outputs.ravel())

        scores = cross_val_score(model, train_inputs, outputs.ravel(), cv=10)
        logging.info("%s | %.02f | %.02f" % (model_name, scores.mean(), scores.std()))


def check_different_data_options():
    inputs, outputs, word_list = pickle.load(open("training_data.pkl", 'rb'))
    # test_inputs, test_outputs, _ = pickle.load(open("testing_data.pkl", 'rb'))
    logging.info("Data loaded")

    tfidf_inputs = TfidfTransformer().fit_transform(inputs)

    vectorizer = TfidfVectorizer(input="filename", decode_error='ignore', strip_accents='ascii', analyzer='word',
                                 ngram_range=(1, 1), stop_words='english', min_df=.1)
    logging.info("Starting...")
    sklearn_inputs = vectorizer.fit_transform(["aclImdb/train/pos/" + x for x in os.listdir("aclImdb/train/pos")] +
                                      ["aclImdb/train/neg/" + x for x in os.listdir("aclImdb/train/neg")])
    logging.info("Finished vectorizing")

    for input_matrix, name in zip([inputs, tfidf_inputs, sklearn_inputs], ['Normal Inputs', 'TF-IDF Inputs', 'Sklearn tfidf']):
        for model, model_name in zip([BernoulliNB(), MultinomialNB(), LogisticRegression(penalty='l1')], ['Bernoulli', 'Multinomial', 'LogReg']):
            scores = cross_val_score(model, input_matrix, outputs.ravel(), cv=10)
            logging.info("Accuracy for %s on %s: %.02f, std: %.02f" % (model_name, name, scores.mean(), scores.std()))


def chi2_cv_feature_selection(inputs, outputs, model, max_features, filename, k=10):
    x_values = np.linspace(1, max_features, num=40, dtype=int)
    kf = KFold(inputs.shape[0], n_folds=k, shuffle=True)
    validation_y_values = []
    training_y_values = []
    for train_indices, val_indices in kf:
        val_y_values = []
        train_y_values = []
        for num_features in x_values:
            feature_sel = SelectKBest(chi2, k=num_features).fit(inputs[train_indices], outputs[train_indices].ravel())

            train_inputs = feature_sel.transform(inputs[train_indices])
            train_outputs = outputs[train_indices].ravel()

            val_inputs = feature_sel.transform(inputs[val_indices])
            val_outputs = outputs[val_indices].ravel()

            model.fit(train_inputs, train_outputs)

            train_y_values.append(accuracy_score(train_outputs, model.predict(train_inputs)))
            val_y_values.append(accuracy_score(val_outputs, model.predict(val_inputs)))

        validation_y_values.append(np.array(val_y_values))
        training_y_values.append(np.array(train_y_values))
        logging.info("Fold finished")
    plt.clf()
    sns.tsplot(data=validation_y_values, time=x_values, ci=[68, 95], color='red', condition="Validation Accuracy")
    sns.tsplot(data=validation_y_values, time=x_values, ci=[68, 95], color='blue', condition="Training Accuracy")

    plt.title("Feature Count vs Feature Count (selected with chi2)")
    plt.xlabel("Feature Count (top k using chi2)")
    plt.ylabel("Accuracy")
    plt.savefig(filename)


def evaluate_baseline():
    inputs, outputs, words = preprocessing.build_data_target_matrices("aclImdb/train/pos", "aclImdb/train/neg", binary_output=True)
    tst_inputs, tst_outputs, _ = preprocessing.build_test_data_target_matrices("aclImdb/test/pos", "aclImdb/test/neg", words, binary_output=True)
    model = BernoulliNB()

    scores = cross_val_score(model, inputs, outputs.ravel(), cv=10)
    logging.info("Accuracy for %s: %.02f, std: %.02f" % ("Baseline BernoulliNB", scores.mean(), scores.std()))

    model.fit(inputs, outputs.ravel())
    logging.info(accuracy_score(tst_outputs.ravel(), model.predict(tst_inputs)))


if __name__ == "__main__":
    utilities.initialize_logger()
    # check_different_data_options()
    main()
