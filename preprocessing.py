import os
import re
import logging
import pickle

import numpy as np
from scipy.sparse import csr_matrix
from collections import Counter

import utilities


class CountDict(dict):
        def increment(self, key):
            self[key] = self.get(key, 0) + 1


def extract_words(line, use_negation=True):
    garbage_symbols = ['<br />']
    for garbage_symbol in garbage_symbols:
        line = line.replace(garbage_symbol, '')
    result = []
    trash_characters = '?.,!:;"$%^&*()#@\'+-/0123456789<>=\\[]_~{}|`'
    for phrase in re.compile('[?.,!:;]').split(line):
        negate_sentence = False
        for word in phrase.split():
            word = word.strip(trash_characters).lower().encode('ascii', 'ignore').decode("ascii")
            if len(word) == 0:
                continue
            if use_negation:
                result.append("not_" + word if negate_sentence else word)
            else:
                result.append(word)

            if word == 'not' or word == 'no' or "n't" in word:
                negate_sentence = True
    return result


def get_document_word_map(directory, limit=-1, n_grams=1):
    result = {}
    count = 0
    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue

        f = open(os.path.join(directory, filename), 'r', encoding='utf-8')
        word_list = extract_words(f.readline())
        n_gram_list = []
        if n_grams > 1:
            for n in range(2, n_grams + 1):
                for i in range(n, len(word_list) + 1):
                    n_gram_list.append("_".join(word_list[i - n: i]))

        result[filename] = word_list + n_gram_list
        f.close()
        count += 1

        if limit != -1 and count >= limit:
            logging.info("Completed reading %d files from %s" % (count, directory))
            return result

    logging.info("Completed reading %d files from %s" % (count, directory))
    return result


def build_data_target_matrices(pos_directory, neg_directory, limit=-1, save_data=False,
                               min_count_thresh=1, binary_output=True,
                               n_grams=1, filename="training_data.pkl"):
    pos_document_word_map = get_document_word_map(pos_directory, limit=limit, n_grams=n_grams)
    neg_document_word_map = get_document_word_map(neg_directory, limit=limit, n_grams=n_grams)
    all_words = CountDict()
    for key in pos_document_word_map.keys():
        [all_words.increment(word) for word in pos_document_word_map[key] if len(word) > 0]
        pos_document_word_map[key] = Counter(pos_document_word_map[key])
    for key in neg_document_word_map.keys():
        [all_words.increment(word) for word in neg_document_word_map[key] if len(word) > 0]
        neg_document_word_map[key] = Counter(neg_document_word_map[key])

    # Build our input/output matrices
    global_word_list = list(sorted([word for word in all_words.keys() if all_words[word] > min_count_thresh]))
    lookup = {word: index for index, word in enumerate(global_word_list)}
    logging.info("Global word list assembled")

    output_matrix = np.zeros((len(pos_document_word_map.keys()) + len(neg_document_word_map.keys()), 1), dtype=np.int8)

    value_list = []
    row_list = []
    col_list = []
    row = 0

    for word_map in [pos_document_word_map, neg_document_word_map]:
        for key in sorted(word_map.keys()):
            words = set(word_map[key])
            for word in words:
                if word in lookup:
                    value_list.append(1 if binary_output else word_map[key][word])
                    row_list.append(row)
                    col_list.append(lookup[word])
            row += 1
    output_matrix[0: len(pos_document_word_map.keys()), ] = 1

    logging.info("Creating sparse input matrix")
    input_matrix = csr_matrix((value_list, (row_list, col_list)), dtype=int)

    if save_data:
        pickle.dump((input_matrix, output_matrix, global_word_list), open(filename, 'wb'))
        logging.info("Data saved successfully!")

    return input_matrix, output_matrix, global_word_list


def build_test_data_target_matrices(pos_directory, neg_directory, train_word_list, save_data=False, binary_output=True):
    logging.info("Beginning to build test data matrices")
    value_list = []
    row_list = []
    col_list = []
    row = 0
    pos_document_word_map = get_document_word_map(pos_directory)
    neg_document_word_map = get_document_word_map(neg_directory)
    for key in pos_document_word_map.keys():
        pos_document_word_map[key] = Counter(pos_document_word_map[key])
    for key in neg_document_word_map.keys():
        neg_document_word_map[key] = Counter(neg_document_word_map[key])

    lookup = {word: index for index, word in enumerate(train_word_list)}
    for directory, word_map in zip([pos_directory, neg_directory], [pos_document_word_map, neg_document_word_map]):
        for filename in os.listdir(directory):
            if not filename.endswith(".txt"):
                continue
            f = open(os.path.join(directory, filename), 'r', encoding='utf-8')
            words = set(extract_words(f.readline()))
            for word in words:
                if word in lookup:
                    value_list.append(1 if binary_output else word_map[filename][word])
                    row_list.append(row)
                    col_list.append(lookup[word])
            row += 1
    logging.info("Sparse data ready")

    output_matrix = np.zeros((len(pos_document_word_map.keys()) + len(neg_document_word_map.keys()), 1), dtype=np.int8)
    output_matrix[0: len(pos_document_word_map.keys()), ] = 1

    logging.info("Creating sparse input matrix")
    input_matrix = csr_matrix((value_list, (row_list, col_list)), dtype=np.int8)

    if save_data:
        pickle.dump((input_matrix, output_matrix, train_word_list), open("testing_data.pkl", 'wb'))
        logging.info("Data saved successfully!")

    return input_matrix, output_matrix, train_word_list
