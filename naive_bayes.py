from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix


if __name__ == "__main__":
    inputs, outputs, word_list = pickle.load(open("training_data.pkl", 'rb'))
    test_inputs, test_outputs, _ = pickle.load(open("testing_data.pkl", 'rb'))
    model = BernoulliNB()
    logging.info("Beginning training")
    model.fit(inputs, outputs.ravel())
    logging.info("Training complete")
    print(confusion_matrix(test_outputs, model.predict(test_inputs)))