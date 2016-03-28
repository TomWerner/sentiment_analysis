import logging

from nose.tools import *
import numpy as np

import preprocessing
import utilities

utilities.initialize_logger()


def test_extract_words():

    text = "This movie, while not great, was decent"
    words = preprocessing.extract_words(text)

    assert_equals(['this', 'movie', 'while', 'not', 'not_great', 'was', 'decent'], words)

    words = preprocessing.extract_words(text, use_negation=False)
    assert_equals(['this', 'movie', 'while', 'not', 'great', 'was', 'decent'], words)


def test_get_document_word_map():
    result = preprocessing.get_document_word_map("../test/pos")
    assert_equals(["test_file_1.txt", "test_file_2.txt"], sorted(result.keys()))

    assert_equals(['this', 'movie', 'while', 'not', 'not_great', 'wasn\'t', 'not_bad'], result['test_file_1.txt'])
    assert_equals(['this', 'movie', 'was', 'amazing'], result['test_file_2.txt'])


def test_build_data_target_matrices():
    input_matrix, output_matrix, word_list = preprocessing.build_data_target_matrices("../test/pos", "../test/neg",
                                                                                      min_count_thresh=0)
    input_matrix = input_matrix.toarray()

    pos_map = preprocessing.get_document_word_map("../test/pos")
    assert_equals(['this', 'movie', 'while', 'not', 'not_great', 'wasn\'t', 'not_bad'], pos_map['test_file_1.txt'])
    assert_equals(['this', 'movie', 'was', 'amazing'], pos_map['test_file_2.txt'])

    neg_map = preprocessing.get_document_word_map("../test/neg")
    assert_equals(['sucked', 'totally', 'sucked'], neg_map['test_file_3.txt'])

    all_words = ['amazing', 'movie', 'not', 'not_bad', 'not_great', 'sucked', 'this', 'totally', 'was', "wasn't", 'while']

    np.testing.assert_array_equal(input_matrix[0], np.array([0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1]))
    np.testing.assert_array_equal(input_matrix[1], np.array([1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0]))
    np.testing.assert_array_equal(input_matrix[2], np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]))

    assert_equals(1, output_matrix[0, 0])
    assert_equals(1, output_matrix[1, 0])
    assert_equals(0, output_matrix[2, 0])

