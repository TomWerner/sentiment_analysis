from preprocessing import build_data_target_matrices, build_test_data_target_matrices
import utilities
import os
import logging

if __name__ == "__main__":
    utilities.initialize_logger()
    logging.info("Program loaded at: %s" % str(os.getcwd()))
    inputs, outputs, word_list = build_data_target_matrices("aclImdb/train/pos/", "aclImdb/train/neg/", save_data=True, binary_output=False)
    # build_test_data_target_matrices("aclImdb/test/pos/", "aclImdb/test/neg/", word_list, save_data=True, binary_output=False)
    logging.info("Finished!")
