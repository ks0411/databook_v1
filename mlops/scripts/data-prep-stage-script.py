import argparse
import os
import pickle

import mlflow
import pandas
from databook.data_book import DataBook
from sklearn.model_selection import train_test_split


def main():

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data (JSON)")
    parser.add_argument("--ranges", type=str,
                        help="path to formula ranges supervision")
    parser.add_argument("--test_train_ratio", type=float,
                        required=False, default=0.25)
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)

    # Load data as a DataBook object
    dBook = DataBook()
    dBook.load_file(args.data)

    # Transform data to features (with negative labels - i.e., vert inconsistent cells not being errors)
    dBook.pre_process_data(for_training=True)

    # Add positive labels (i.e., vert inconsistent cells being errors)
    dBook.add_positive_cases(args.ranges)

    # Cast to a Pandas df
    df = dBook.get_data(all_columns=False).copy(deep=True)
    df.reset_index(inplace=True)

    # Identify sheets with positive labels for training
    sheet_names = df[df.Label == True]['sheetName'].unique()
    red_df = df[df['sheetName'].isin(sheet_names)]

    # Populate data (X) and labels (Y) and split
    features = [c for c in df.columns if c not in [
        'Label', 'key', 'cellAddress', 'sheetName']]
    X = red_df[features]
    Y = red_df['Label']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)

    y_train_count = Y_train.value_counts(normalize=False)
    y_test_count = Y_test.value_counts(normalize=False)

    print(
        f"Positive/Negative labels count in train: {y_train_count.loc[True]}/{y_train_count.loc[False]}")
    print(
        f"Positive/Negative labels count in test: {y_test_count.loc[True]}/{y_test_count.loc[False]}")

    mlflow.log_metric("num_positive_samples (train)", y_train_count.loc[True])
    mlflow.log_metric("num_negative_samples (train)", y_train_count.loc[False])
    mlflow.log_metric("num_positive_samples (test)", y_test_count.loc[True])
    mlflow.log_metric("num_negative_samples (test)", y_test_count.loc[False])

    os.makedirs(args.train_data, exist_ok=True)
    file_name_train = os.path.join(args.train_data, "train.pkl")
    train_data = {"X_train": X_train, "Y_train": Y_train}

    with open(file_name_train, "wb") as f_train:
        pickle.dump(train_data, f_train)

    os.makedirs(args.test_data, exist_ok=True)
    file_name_test = os.path.join(".", args.test_data, "test.pkl")
    test_data = {"X_test": X_test, "Y_test": Y_test}

    with open(file_name_test, "wb") as f_test:
        pickle.dump(test_data, f_test)

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
