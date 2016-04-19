import pickle
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
import logging
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
import os
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

import utilities
from collections import Counter
import preprocessing
import numpy as np
from scipy.sparse import csr_matrix
from flask import Flask, render_template, request, Response, send_from_directory, jsonify, escape
import random

app = Flask(__name__)
pipeline = None
word_list = None


@app.route("/try_your_own", methods=['POST'])
def try_your_own():
    review_text = request.get_json(force=True)['text']
    return jsonify({'result': str(decide(review_text)[0]), 'text': escape(review_text)})


@app.route("/rand_rev", methods=['POST'])
def rand_rev():
    which_type = request.get_json(force=True)['type']
    directory = "aclImdb/test/" + ('pos' if which_type == 'pos' else 'neg')
    print(directory )
    filename = random.choice(os.listdir(directory))
    review_text = open(os.path.join(directory, filename)).readline()

    return jsonify({'result': str(decide(review_text)[0]), 'text': "Expected: " + which_type + "<br/>" + review_text})


@app.route("/")
def home():
    return render_template('home.html')


def load_model():
    if os.path.isfile("demo_model.pkl"):
        pipeline, word_list = pickle.load(open("demo_model.pkl", 'rb'))
        logging.info("Model loaded")
        return pipeline, word_list
    else:
        inputs, outputs, word_list = pickle.load(open("training_data_3_gram.pkl", 'rb'))
        logging.info("Data loaded")
        pipeline = Pipeline([('tfidf', TfidfTransformer()),
                             ('chi2_top_k', SelectKBest(chi2, 13000)),
                             ('l1_step', SelectFromModel(LinearSVC(penalty='l1', dual=False, C=1))),
                             ('linear_svc', LinearSVC(penalty='l2', C=1))])
        pipeline.fit(inputs, outputs.ravel())
        logging.info("Pipeline trained")

        pickle.dump((pipeline, word_list), open("demo_model.pkl", 'wb'))
        logging.info("Pipeline saved")
        return load_model()


def decide(text):
    dictionary = {}
    preprocessing.handle_file_text("demo", text, 3, dictionary)
    value_list = []
    row_list = []
    col_list = []
    row = 0

    for key in dictionary.keys():
        dictionary[key] = Counter(dictionary[key])
    lookup = {word: index for index, word in enumerate(word_list)}

    words = set(preprocessing.extract_words(text))
    for word in words:
        if word in lookup:
            value_list.append(dictionary['demo'][word])
            row_list.append(row)
            col_list.append(lookup[word])
    row += 1
    input_matrix = csr_matrix((value_list, (row_list, col_list)), shape=(1, len(word_list)), dtype=np.int8)

    return pipeline.predict(input_matrix)


if __name__ == "__main__":
    utilities.initialize_logger()
    pipeline, word_list = load_model()
    app.run(threaded=False)
