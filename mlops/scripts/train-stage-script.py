import argparse
import os
import pickle

import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
# import pandas
# from databook.data_book import DataBook
from sklearn.model_selection import train_test_split


def main():

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    parser.add_argument("--model_path", type=str, help="path to the model")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    print("train data path:", args.train_data)
    print("test data path:", args.test_data)

    file_name_train = os.path.join(args.train_data, "train.pkl")

    with open(file_name_train, "rb") as f_train:
        train_data = pickle.load(f_train)

    file_name_test = os.path.join(args.test_data, "test.pkl")

    with open(file_name_test, "rb") as f_test:
        test_data = pickle.load(f_test)

    LRC = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')
    LRC.fit(train_data["X_train"], train_data["Y_train"])

    predicted_LRC = LRC.predict(test_data["X_test"])
    tn, fp, fn, tp = confusion_matrix(
        predicted_LRC, test_data["Y_test"]).ravel()
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)

    mlflow.log_metric("TP (test)", tp)
    mlflow.log_metric("TN (test)", tn)
    mlflow.log_metric("FP (test)", fp)
    mlflow.log_metric("FN (test)", fn)
    mlflow.log_metric("Accuracy (test)", accuracy)
    mlflow.log_metric("Precision (test)", precision)
    mlflow.log_metric("Recall (test)", recall)

    print(classification_report(test_data["Y_test"], predicted_LRC))

    os.makedirs(args.model_path, exist_ok=True)
    file_name_model = os.path.join(args.model_path, "model.pkl")

    with open(file_name_model, "wb") as f_model:
        pickle.dump(LRC, f_model)

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
