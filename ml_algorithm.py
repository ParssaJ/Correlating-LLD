import time
from ast import literal_eval
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


def training_data_df():
    training_data = pd.read_csv("csv_files/training_data_with_parsed_content.csv")
    training_data = training_data[["Webdomains", "guessed-domain", "website_content"]]
    training_data["website_content"] = training_data["website_content"].apply(literal_eval)

    mlb = MultiLabelBinarizer()
    content_ohe = pd.DataFrame(mlb.fit_transform(training_data["website_content"]), columns=mlb.classes_)

    le = LabelEncoder()
    domains_labeled = pd.DataFrame(le.fit_transform(training_data["guessed-domain"]))

    result_df = pd.concat([domains_labeled, content_ohe], axis=1)
    training_data.reset_index(inplace=True, drop=True)

    return result_df


if __name__ == '__main__':
    start = time.time()

    df = training_data_df()
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    forest = RandomForestClassifier(random_state=0, n_jobs=-1)
    forest.fit(X_train, y_train)

    predicted = forest.predict(X_test)
    print(f"Accuracy with stock-RandomForestTreeClassifier: {accuracy_score(predicted, y_test)}")

    df = df[:2000]
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    ovr = OneVsRestClassifier(estimator=SVC(random_state=0), n_jobs=-1)
    ovr.fit(X_train, y_train)

    predicted = ovr.predict(X_test)
    print(f"Accuary with OneVsRest SVM(rbf): {accuracy_score(predicted, y_test)}")

    end = time.time()
    print(f"Took a total of {np.round((end - start) / 60, 2)} minutes")
