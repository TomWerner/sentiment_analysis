import os
import re
import logging
import pickle

import numpy as np
from scipy.sparse import csr_matrix

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

            if use_negation:
                result.append("not_" + word if negate_sentence else word)
            else:
                result.append(word)

            if word == 'not' or word == 'no' or "n't" in word:
                negate_sentence = True
    return result


def get_document_word_map(directory, limit=-1):
    result = {}
    count = 0
    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue

        f = open(os.path.join(directory, filename), 'r', encoding='utf-8')
        result[filename] = extract_words(f.readline())
        f.close()
        count += 1

        if limit != -1 and count >= limit:
            logging.info("Completed reading %d files from %s" % (count, directory))
            return result

    logging.info("Completed reading %d files from %s" % (count, directory))
    return result


def build_data_target_matrices(pos_directory, neg_directory, limit=-1, save_data=False, min_count_thresh=1):
    pos_document_word_map = get_document_word_map(pos_directory, limit=limit)
    neg_document_word_map = get_document_word_map(neg_directory, limit=limit)
    all_words = CountDict()
    for key in pos_document_word_map.keys():
        [all_words.increment(word) for word in pos_document_word_map[key] if len(word) > 0]
    for key in neg_document_word_map.keys():
        [all_words.increment(word) for word in neg_document_word_map[key] if len(word) > 0]

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
                    value_list.append(1)
                    row_list.append(row)
                    col_list.append(lookup[word])
            row += 1
    output_matrix[0: len(pos_document_word_map.keys()), ] = 1

    logging.info("Creating sparse input matrix")
    input_matrix = csr_matrix((value_list, (row_list, col_list)), dtype=np.int8)

    if save_data:
        pickle.dump((input_matrix, output_matrix, global_word_list), open("training_data.pkl", 'wb'))
        logging.info("Data saved successfully!")

    return input_matrix, output_matrix, global_word_list


def build_test_data_target_matrices(pos_directory, neg_directory, train_word_list):
    logging.info("Beginning to build test data matrices")
    value_list = []
    row_list = []
    col_list = []
    row = 0

    lookup = {word: index for index, word in enumerate(train_word_list)}
    for directory in [pos_directory, neg_directory]:
        for filename in os.listdir(directory):
            if not filename.endswith(".txt"):
                continue
            f = open(os.path.join(directory, filename), 'r', encoding='utf-8')
            words = set(extract_words(f.readline()))
            for word in words:
                if word in lookup:
                    value_list.append(1)
                    row_list.append(row)
                    col_list.append(lookup[word])
            row += 1
    logging.info("Sparse data ready")
    pos_document_word_map = get_document_word_map(pos_directory)
    neg_document_word_map = get_document_word_map(neg_directory)

    output_matrix = np.zeros((len(pos_document_word_map.keys()) + len(neg_document_word_map.keys()), 1), dtype=np.int8)
    output_matrix[0: len(pos_document_word_map.keys()), ] = 1

    logging.info("Creating sparse input matrix")
    input_matrix = csr_matrix((value_list, (row_list, col_list)), dtype=np.int8)

    pickle.dump((input_matrix, output_matrix, train_word_list), open("testing_data.pkl", 'wb'))
    logging.info("Data saved successfully!")
